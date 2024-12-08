import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb 
import uuid  # For generating unique run names
import time
from crossmodal_training import (
    MultiModalFeatureExtractor,
    create_voxceleb2_dataloader
)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch
    
    Args:
        model (nn.Module): The multi-modal feature extractor
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Computing device
    
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # Progress bar for training
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        frames = batch['frames'].to(device)
        mel_spec = batch['mel_spectrogram'].to(device)
        audio_lengths = batch['audio_length'].to(device)
        frame_times = batch['frame_times'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        combined_features = model(
            frames,
            mel_spec,
            audio_lengths,
            frame_times
        )
        
        # Compute loss (you'll need to define an appropriate loss function)
        # This is a placeholder - replace with your specific loss computation
        loss = criterion(combined_features, combined_features)  # Placeholder
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return {
        'loss': avg_loss
    }

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model (nn.Module): The multi-modal feature extractor
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Computing device
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    # Progress bar for validation
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            frames = batch['frames'].to(device)
            mel_spec = batch['mel_spectrogram'].to(device)
            audio_lengths = batch['audio_length'].to(device)
            frame_times = batch['frame_times'].to(device)
            
            # Forward pass
            combined_features = model(
                frames,
                mel_spec,
                audio_lengths,
                frame_times
            )
            
            # Compute loss (placeholder)
            loss = criterion(combined_features, combined_features)  # Placeholder
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': loss.item()})
    
    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    return {
        'val_loss': avg_loss
    }

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    dataset_path = Path(r"/work/scratch/kurse/kurs00079/data/vox2")
    
    train_loader = create_voxceleb2_dataloader(
        root_dir=dataset_path,
        batch_size=config["batch_size"],
        num_workers=0,
        split='dev',
        frames_per_clip=25,
        frame_size=(112, 112),
        max_audio_length=30
    )
    
    val_loader = create_voxceleb2_dataloader(
        root_dir=dataset_path,
        batch_size=2,
        num_workers=0,
        split='dev',
        frames_per_clip=25,
        frame_size=(112, 112),
        max_audio_length=30
    )
    
    # Training loop
    print("Starting training...")
    
    # Tracking metrics
    training_history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Main training loop
    for epoch in range(10):  # 10 epochs as requested
        print(f"\nEpoch {epoch + 1}/10")
        start_time = time.time()
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_model(model, val_loader, criterion, device)
        epoch_duration = time.time() - start_time
        # Store metrics
        training_history['train_loss'].append(train_metrics['loss'])
        training_history['val_loss'].append(val_metrics['val_loss'])
        
        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['val_loss']
        }, checkpoint_path)
        
        # Log metrics and checkpoint to wandb
        wandb.log({
            "train_loss": train_metrics['loss'],
            "val_loss": val_metrics['val_loss'],
            "epoch_duration": epoch_duration,
            "checkpoint": wandb.Artifact(
                name=f"checkpoint-epoch-{epoch+1}", 
                type="model",
                metadata={
                    "epoch": epoch+1,
                    "train_loss": train_metrics['loss'],
                    "val_loss": val_metrics['val_loss']
                }
            )
        })
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    wandb.log({"training_loss_plot": wandb.Image('training_loss_plot.png')})
    
    wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise