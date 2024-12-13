import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb 
import uuid  
import time
import gc
import os
from loss_function import cross_modal_consistency_loss
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from crossmodal_training import (
    MultiModalFeatureExtractor,
    create_voxceleb2_dataloader
)

class MemoryTracker:
    def __init__(self, tracking_points):
        """
        Initialize memory tracker with specified tracking points
        
        tracking_points: list of strings describing when memory is tracked
        """
        self.tracking_points = tracking_points
        self.memory_data = {point: [] for point in tracking_points}
        
    def print_gpu_tensors(self):
        """Print out all tensors currently in GPU memory with their sizes."""
        for obj in gc.get_objects():
            try:
                if obj.element_size() * obj.nelement()  / 1024 ** 2 > 10:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        if obj.device.type == 'cuda':
                            print(f"Tensor type: {type(obj)}")
                            print(f"Size: {obj.size()}")
                            print(f"Memory: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
                            print("---")
            except:
                pass

    
    def log_memory(self, point, device):
        """
        Log memory for a specific tracking point
        """
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        
        self.memory_data[point].append({
            'reserved': reserved_memory,
            'allocated': allocated_memory
        })
    
    def plot_memory_across_epochs(self, output_dir='memory_plots'):
        """
        Create plots for memory usage across epochs for each tracking point
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for point in self.tracking_points:
            # Prepare data for plotting
            reserved_data = [entry['reserved'] for entry in self.memory_data[point]]
            allocated_data = [entry['allocated'] for entry in self.memory_data[point]]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot reserved memory
            plt.plot(reserved_data, label='Reserved Memory (GB)', color='blue', marker='o')
            plt.plot(allocated_data, label='Allocated Memory (GB)', color='red', marker='x')
            
            plt.title(f'Memory Usage at {point}')
            plt.xlabel('Epoch')
            plt.ylabel('Memory (GB)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save and close plot
            plot_filename = os.path.join(output_dir, f'{point}_memory_plot.png')
            plt.savefig(plot_filename)
            plt.close()


def train_epoch(model: MultiModalFeatureExtractor, train_loader, criterion, optimizer, device, memory_tracker, current_epoch):
    """
    Train the model for one epoch with memory tracking
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # Initial memory tracking
    memory_tracker.log_memory('start_of_epoch', device)
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        if not batch['valid'].all():
            print("Skipping Batch due to invalid shape")
        else:
            try:
                # Move batch to device
                frames = batch['frames'].to(device)
                memory_tracker.log_memory('after_batch_to_device', device)
                
                mel_spec = batch['mel_spectrogram'].to(device)
                audio_lengths = batch['audio_length'].to(device)
                frame_times = batch['frame_times'].to(device)
                
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                memory_tracker.log_memory('after_zero_grad', device)
                
                # Forward pass
                visual_features, audio_feautres = model(
                    frames,
                    mel_spec,
                    audio_lengths,
                    frame_times
                )
                memory_tracker.log_memory('after_forward_pass', device)
                
                # Compute loss (real)
                loss = criterion(visual_features, audio_feautres,0.7)  
                
                # Backward pass and optimize
                loss.backward()
                memory_tracker.log_memory('after_backward_pass', device)
                
                optimizer.step()
                memory_tracker.log_memory('after_optimizer_step', device)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})


            except RuntimeError as e:
                print(f"Error processing batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                raise

    # End of epoch memory tracking
    memory_tracker.log_memory('end_of_epoch', device)
    
    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return {
        'loss': avg_loss
    }

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    use_wandb = True
    # Tracking points
    tracking_points = [
        'start_of_epoch', 
        'after_batch_to_device', 
        'after_zero_grad', 
        'after_forward_pass', 
        'after_backward_pass', 
        'after_optimizer_step', 
        'end_of_epoch'
    ]
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker(tracking_points)
    
    # Generate a unique experiment name and run name
    experiment_name = "multimodal-feature-extractor-v1"
    run_name = f"{experiment_name}-{uuid.uuid4().hex[:8]}"
    
    config = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
    }
    
    # Initialize wandb with experiment and run names
    if use_wandb:
        wandb.init(
            project=experiment_name, 
            name=run_name,
            config=config
        )
    
    model = MultiModalFeatureExtractor( 
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"]
    ).to(device)
    
    criterion = cross_modal_consistency_loss 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    dataset_path = Path(r"/work/scratch/kurse/kurs00079/data/vox2")
    #dataset_path = Path(r"D:\dataset")
    train_loader = create_voxceleb2_dataloader(
        root_dir=dataset_path,
        batch_size=config["batch_size"],
        num_workers=1,
        split='dev',
        max_video_length=25,
        frame_size=(112, 112),
        max_audio_length=30,
        goal_fps=25,
        max_videos= 10000
    )
    
    # Training loop
    print("Starting training...")
    
    # Tracking metrics
    training_history = {
        'train_loss': [],
    }
    
    # Main training loop
    for epoch in range(10):  # 10 epochs as requested
        print(f"\nEpoch {epoch + 1}/10")
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            memory_tracker, 
            epoch
        )
        
        epoch_duration = time.time() - start_time
        
        # Store metrics
        training_history['train_loss'].append(train_metrics['loss'])
        
        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
        }, checkpoint_path)
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "epoch_duration": epoch_duration,
            })
        
    # Create memory usage plots across epochs
    memory_tracker.plot_memory_across_epochs()
    if use_wandb:
        wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise