import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from crossmodal_training import (
    MultiModalFeatureExtractor,
    create_voxceleb2_dataloader
)

def visualize_batch(batch):
    """Visualize the first frame from each video in the batch"""
    frames = batch['frames']
    
    # Denormalize the images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = frames * std + mean
    
    # Plot the first frame from each video
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    for i, video in enumerate(frames):
        frame = video[0].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        if len(frames) > 1:
            axes[i].imshow(frame)
            axes[i].axis('off')
            axes[i].set_title(f'Video {i+1}')
        else:
            axes.imshow(frame)
            axes.axis('off')
            axes.set_title('Video 1')
    
    plt.tight_layout()
    plt.show()

def visualize_mel_spectrograms(batch):
    """Visualize mel spectrograms from the batch"""
    mel_specs = batch['mel_spectrogram']
    
    fig, axes = plt.subplots(1, len(mel_specs), figsize=(15, 5))
    for i, mel in enumerate(mel_specs):
        if len(mel_specs) > 1:
            axes[i].imshow(mel[0], aspect='auto', origin='lower')
            axes[i].set_title(f'Mel Spectrogram {i+1}')
        else:
            axes.imshow(mel[0], aspect='auto', origin='lower')
            axes.set_title('Mel Spectrogram 1')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing MultiModal Feature Extractor...")
    model = MultiModalFeatureExtractor().to(device)
    model.eval()  # Set to evaluation mode
    
    # Create dataloader
    dataset_path = Path(r"E:\MAI\git2\Project-Lab-MAI\datasets")
    print(f"Creating dataloader from: {dataset_path}")
    
    dataloader = create_voxceleb2_dataloader(
        root_dir=dataset_path,
        batch_size=2,
        num_workers=0,  # Set to 0 for easier debugging
        split='test',
        frames_per_clip=16,
        frame_size=(112, 112),
        max_audio_length=30
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Process a few batches
    print("\nProcessing batches...")
    with torch.no_grad():  # Disable gradient computation for inference
        for batch_idx, batch in enumerate(tqdm(dataloader, total=min(3, len(dataloader)))):
            if batch_idx >= 3:  # Only process first 3 batches as example
                break
                
            # Print batch information
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Number of videos: {len(batch['frames'])}")
            print(f"Frames shape: {batch['frames'].shape}")
            print(f"Mel spectrogram shape: {batch['mel_spectrogram'].shape}")
            print(f"Audio lengths: {batch['audio_length']}")
            print(f"Frame timestamps shape: {batch['frame_times'].shape}")
            
            # Visualize the data
            print("\nVisualizing first frames from videos...")
            #visualize_batch(batch)
            
            print("\nVisualizing mel spectrograms...")
            #visualize_mel_spectrograms(batch)
            
            # Move batch to device
            frames = batch['frames'].to(device)
            mel_spec = batch['mel_spectrogram'].to(device)
            audio_lengths = batch['audio_length'].to(device)
            frame_times = batch['frame_times'].to(device)
            
            # Forward pass
            print("\nPerforming forward pass...")
            combined_features = model(
                frames,
                mel_spec,
                audio_lengths,
                frame_times
            )
            
            # Print feature information
            print(f"Combined features shape: {combined_features.shape}")
            print(f"Feature statistics:")
            print(f"  Mean: {combined_features.mean().item():.4f}")
            print(f"  Std: {combined_features.std().item():.4f}")
            print(f"  Min: {combined_features.min().item():.4f}")
            print(f"  Max: {combined_features.max().item():.4f}")
            
            # Visualize feature distributions
            plt.figure(figsize=(10, 5))
            plt.hist(combined_features.cpu().numpy().flatten(), bins=50)
            plt.title('Distribution of Combined Features')
            plt.xlabel('Feature Value')
            plt.ylabel('Count')
            plt.show()
            
            print("\n" + "="*50 + "\n")
            
            
            
            
            
            




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise