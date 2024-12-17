import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
from crossmodal_training import MultiModalFeatureExtractor
from loss_function import cross_modal_consistency_loss
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import os
import whisper
import torchvision.transforms as transforms
from torchvision.io import read_video, read_video_timestamps
from PIL import Image
import json
from typing import Dict, Any

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def save_results_with_numpy_support(data, filename):
    """
    Save results with proper handling of numpy types
    
    :param data: Dictionary or list to be saved
    :param filename: Output JSON filename
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

def print_detailed_results(test_results: Dict[str, Any], output_dir: str = 'test_results'):
    """
    Print and save comprehensive test results with threshold-based breakdown
    
    :param test_results: Results dictionary from test_model_with_auc
    :param output_dir: Directory to save result files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare a comprehensive results summary
    comprehensive_summary = {}
    
    # Iterate through thresholds
    for threshold_key, threshold_data in test_results['overall'].items():
        print(f"\n--- Results for {threshold_key} ---")
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  ROC AUC: {threshold_data['roc_auc']:.4f}")
        print(f"  Average Precision: {threshold_data['avg_precision']:.4f}")
        
        # Per-type classification results
        print("\nType-Specific Classification Report:")
        type_classification = test_results.get('type_classification', {})
        
        # Prepare threshold-specific summary
        threshold_summary = {
            'overall_metrics': {
                'roc_auc': threshold_data['roc_auc'],
                'avg_precision': threshold_data['avg_precision']
            },
            'type_metrics': {}
        }
        
        # Iterate through types
        for type_name, type_data in type_classification.items():
            print(f"\n{type_name}:")
            print(f"  Total Samples: {type_data['total_samples']}")
            print(f"  Correct Classifications: {type_data['correct_classifications']}")
            print(f"  Incorrect Classifications: {type_data['incorrect_classifications']}")
            print(f"  Accuracy: {type_data['accuracy']:.4f}")
            
            print("  Confusion Matrix:")
            cm = type_data['confusion_matrix']
            print(f"    True Positives: {cm['true_positives']}")
            print(f"    True Negatives: {cm['true_negatives']}")
            print(f"    False Positives: {cm['false_positives']}")
            print(f"    False Negatives: {cm['false_negatives']}")
            
            # Calculate additional metrics
            precision = (cm['true_positives'] / (cm['true_positives'] + cm['false_positives'])) if (cm['true_positives'] + cm['false_positives']) > 0 else 0
            recall = (cm['true_positives'] / (cm['true_positives'] + cm['false_negatives'])) if (cm['true_positives'] + cm['false_negatives']) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
            
            # Store in threshold summary
            threshold_summary['type_metrics'][type_name] = {
                'total_samples': type_data['total_samples'],
                'accuracy': type_data['accuracy'],
                'confusion_matrix': cm,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        
        # Save results to JSON file
        output_file = os.path.join(output_dir, f'test_results_{threshold_key}.json')
        save_results_with_numpy_support(threshold_summary, output_file)
        
        print(f"\nDetailed results saved to {output_file}")
        
        # Accumulate in comprehensive summary
        comprehensive_summary[threshold_key] = threshold_summary
    
    # Save comprehensive summary
    comprehensive_output_file = os.path.join(output_dir, 'comprehensive_test_results.json')
    save_results_with_numpy_support(comprehensive_summary, output_file)
    
    print(f"\nComprehensive results saved to {comprehensive_output_file}")
    

class LavDFDataset(Dataset):
    def __init__(
        self, 
        metadata_path, 
        root_dir, 
        frame_size=(112, 112), 
        max_video_length=30, 
        max_audio_length=30, 
        goal_fps=1,
        n_mels=128,
        label_method='modify_check'
    ):
        """
        Initialize LAV-DF Dataset with JSON metadata
        
        :param metadata_path: Path to the JSON metadata file
        :param root_dir: Root directory containing video files
        :param frame_size: Target frame size for video processing
        :param max_video_length: Maximum video length in seconds
        :param max_audio_length: Maximum audio length in seconds
        :param goal_fps: Target frames per second
        :param n_mels: Number of mel spectrogram mel bands
        :param label_method: Method to assign labels ('modify_check', 'fake_periods', or 'n_fakes')
        """
        random.seed(0)
        np.random.seed(0)
        # Load JSON metadata
        with open(metadata_path, 'r') as f:
            fullmetadata = json.load(f)
        test_metadata = [
        entry for entry in fullmetadata
        if entry['file'][:4] == "test"
        ]
        self.metadata = random.sample(test_metadata, min(len(test_metadata), 2000))
        self.root_dir = Path(root_dir)
        self.frame_size = frame_size
        self.max_video_length = max_video_length
        self.max_audio_length = max_audio_length
        self.goal_fps = goal_fps
        self.n_mels = n_mels
        self.label_method = label_method
        
        # Whisper for mel spectrograms
        self.whisper_processor = whisper.log_mel_spectrogram
        
        # Video transforms
        self.video_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _assign_label(self, metadata_entry):
        """
        Assign a label based on the specified method
        
        :param metadata_entry: Dictionary containing metadata for a single video
        :return: Binary label (0 for real, 1 for fake)
        """
        if self.label_method == 'modify_check':
            # Label based on video or audio modification
            return int(metadata_entry['modify_video'] or metadata_entry['modify_audio']),  metadata_entry['modify_video'] + metadata_entry['modify_audio']* 2 
        
        elif self.label_method == 'fake_periods':
            # Label based on presence of fake periods
            return int(len(metadata_entry['fake_periods']) > 0)
        
        elif self.label_method == 'n_fakes':
            # Label based on number of fakes
            return int(metadata_entry['n_fakes'] > 0)
        
        else:
            raise ValueError(f"Unknown label method: {self.label_method}")
    
    def _process_frame(self, frame):
        """Convert frame to PIL, resize, and transform to input tensor."""
        pil_image = Image.fromarray(frame.numpy().transpose(1, 2, 0).astype(np.uint8))
        
        # Resize to target frame size
        resized_image = pil_image.resize(self.frame_size)
        frame_tensor = transforms.ToTensor()(resized_image)
        return frame_tensor
    
    def _load_video_frames(self, video_path):
        """Load and process video frames."""
        try:
            pts, fps = read_video_timestamps(str(video_path))
            frames, audio, info = read_video(
                str(video_path),
                output_format="TCHW"
            )

            # Optional frame rate reduction
            if self.goal_fps and self.goal_fps < info['video_fps']:
                skip_interval = max(1, int(round(info['video_fps'] / self.goal_fps)))
                frames = frames[::skip_interval]
                pts = pts[::skip_interval]

            total_frames = int(self.max_video_length * (self.goal_fps or info['video_fps']))
            
            # Trim or pad frames
            if frames.size(0) > total_frames:
                frames = frames[:total_frames]
                pts = pts[:total_frames]
            
            processed_frames = torch.stack([self._process_frame(frame) for frame in frames])
            processed_frames = self.video_transforms(processed_frames)
            valid = True

            # Pad with black frames if needed
            padding_frames = total_frames - processed_frames.size(0)
            if padding_frames > 0:
                black_frame = torch.zeros_like(processed_frames[0])
                processed_frames = torch.cat(
                    [processed_frames, black_frame.repeat(padding_frames, 1, 1, 1)],
                    dim=0
                )

            # Pad pts (timestamps)
            if len(pts) > 1:
                interval = pts[1] - pts[0]
                last_pt = pts[-1]
                additional_pts = [last_pt + interval * (i + 1) for i in range(padding_frames)]
                pts = torch.cat([torch.tensor(pts), torch.tensor(additional_pts)])
            else:
                pts = torch.tensor(pts)

            return processed_frames, audio, pts, valid
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return dummy data on error
            total_frames = int(self.max_video_length * self.goal_fps)
            dummy_frames = torch.zeros((total_frames, 3, *self.frame_size))
            dummy_audio = torch.zeros(1, 16000 * self.max_audio_length)
            dummy_pts = torch.arange(total_frames).float()
            return dummy_frames, dummy_audio, dummy_pts, False

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        video_path = self.root_dir / entry['file']
        label, type = self._assign_label(entry)

        frames, audio, pts, valid = self._load_video_frames(video_path)
        
        audio_len = audio.size(1) / 16000
        audio = whisper.pad_or_trim(audio)
        mel = self.whisper_processor(audio.squeeze(1).numpy(), n_mels=self.n_mels).squeeze(0)

        return {
            "frames": frames,
            "mel_spectrogram": mel,
            "audio_length": audio_len,
            "video_path": str(video_path),
            "frame_times": pts,
            "label": label,
            "valid": valid,
            "type": type,
        }

def create_lav_df_dataloader(
    metadata_path,
    root_dir,
    batch_size=8,
    num_workers=0,
    frame_size=(112, 112),
    max_video_length=30,
    max_audio_length=30,
    goal_fps=1,
    label_method='modify_check'
):
    """
    Create a DataLoader for the LAV-DF dataset
    
    :param metadata_path: Path to JSON metadata file
    :param root_dir: Root directory containing videos
    :param batch_size: Number of samples per batch
    :param num_workers: Number of subprocesses for data loading
    :param frame_size: Target frame size
    :param max_video_length: Maximum video length in seconds
    :param max_audio_length: Maximum audio length in seconds
    :param goal_fps: Target frames per second
    :param label_method: Method to assign labels
    :return: PyTorch DataLoader
    """
    dataset = LavDFDataset(
        metadata_path=metadata_path,
        root_dir=root_dir,
        frame_size=frame_size,
        max_video_length=max_video_length,
        max_audio_length=max_audio_length,
        goal_fps=goal_fps,
        label_method=label_method
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Set to False for testing
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader

def test_model_with_auc(
    model, 
    test_loader, 
    device, 
    criterion,
    thresholds=[2, 4, 6],
    use_wandb=False
):
    """
    Comprehensive testing method with AUC and multiple threshold evaluations,
    including type-based metrics and classification breakdown
    
    :param model: Trained MultiModalFeatureExtractor
    :param test_loader: DataLoader for test dataset
    :param device: Computation device
    :param criterion: Cross-modal consistency loss function
    :param thresholds: List of thresholds to evaluate
    :param use_wandb: Whether to log results to Weights & Biases
    :return: Dictionary of comprehensive test metrics
    """
    model.eval()
    
    # Detailed storage for analysis
    all_losses = []
    all_labels = []
    all_types = []
    
    # Collect losses, labels, and types
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Collecting Losses", leave=False)
        
        for batch in progress_bar:
            if not batch['valid'].all():
                continue

            # Move data to device
            frames = batch['frames'].to(device)
            mel_spec = batch['mel_spectrogram'].to(device)
            audio_lengths = batch['audio_length'].to(device)
            frame_times = batch['frame_times'].to(device)
            labels = batch['label'].to(device)
            types = batch['type'].to(device)

            # Forward pass
            visual_features, audio_features = model(
                frames, 
                mel_spec, 
                audio_lengths, 
                frame_times
            )

            # Compute loss
            loss = criterion(visual_features, audio_features, 0.7)
            
            # Store results
            all_losses.append(loss.cpu().item())
            all_labels.extend(labels.cpu().numpy())
            all_types.extend(types.cpu().numpy())

    # Convert to numpy arrays
    all_losses = np.array(all_losses)
    all_labels = np.array(all_labels)
    all_types = np.array(all_types)
    
    # Results dictionary to store metrics
    results = {
        'overall': {},
        'types': {}
    }
    
    # Plot ROC and Precision-Recall curves
    plt.figure(figsize=(20, 5))
    
    # Type mapping
    type_mapping = {
        0: 'No Modification',
        1: 'Video Modification',
        2: 'Audio Modification',
        3: 'Both Video and Audio Modification'
    }
    
    # Detailed analysis for each type and overall
    for threshold in thresholds:
        # Overall metrics
        overall_predictions = (all_losses > threshold).astype(int)
        
        # Compute overall metrics
        overall_roc_auc = roc_auc_score(all_labels, all_losses)
        overall_avg_precision = average_precision_score(all_labels, all_losses)
        
        results['overall'][f'threshold_{threshold}'] = {
            'roc_auc': overall_roc_auc,
            'avg_precision': overall_avg_precision,
            'predictions': overall_predictions
        }
        
        # Type-specific analysis
        type_metrics = {}
        type_classification_report = {}
        
        for type_val in np.unique(all_types):
            # Filter for specific type
            type_mask = all_types == type_val
            type_losses = all_losses[type_mask]
            type_labels = all_labels[type_mask]
            
            # Skip if no samples for this type
            if len(type_labels) == 0:
                continue
            
            # Predictions for this type
            type_predictions = (type_losses > threshold).astype(int)
            
            # Compute type-specific metrics
            try:
                type_roc_auc = roc_auc_score(type_labels, type_losses)
                type_avg_precision = average_precision_score(type_labels, type_losses)
            except ValueError:
                # Handle cases with only one class
                type_roc_auc = 0
                type_avg_precision = 0
            
            # Detailed classification report
            type_classification = {
                'total_samples': len(type_labels),
                'correct_classifications': np.sum(type_predictions == type_labels),
                'incorrect_classifications': np.sum(type_predictions != type_labels),
                'accuracy': np.mean(type_predictions == type_labels),
                'confusion_matrix': {
                    'true_positives': np.sum((type_predictions == 1) & (type_labels == 1)),
                    'true_negatives': np.sum((type_predictions == 0) & (type_labels == 0)),
                    'false_positives': np.sum((type_predictions == 1) & (type_labels == 0)),
                    'false_negatives': np.sum((type_predictions == 0) & (type_labels == 1))
                }
            }
            
            # Store metrics
            type_metrics[type_mapping[type_val]] = {
                'roc_auc': type_roc_auc,
                'avg_precision': type_avg_precision,
                'predictions': type_predictions
            }
            
            type_classification_report[type_mapping[type_val]] = type_classification
        
        # Store type-specific metrics
        results['types'][f'threshold_{threshold}'] = type_metrics
        results['type_classification'] = type_classification_report
        
        # Plotting
        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(all_labels, all_losses)
        plt.plot(fpr, tpr, label=f'Threshold {threshold} (Overall AUC = {overall_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 2)
        precision, recall, _ = precision_recall_curve(all_labels, all_losses)
        plt.plot(recall, precision, label=f'Threshold {threshold} (AP = {overall_avg_precision:.2f})')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
    
    # Distribution of losses
    plt.subplot(1, 3, 3)
    plt.hist([all_losses[all_labels == 0], all_losses[all_labels == 1]], 
             label=['Real', 'Fake'], bins=50, alpha=0.7)
    plt.title('Loss Distribution')
    plt.xlabel('Cross-Modal Consistency Loss')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('deepfake_detection_analysis.png')
    plt.close()
    
    # Optional Weights & Biases logging
    if use_wandb:
        import wandb
        for threshold, metrics in results['overall'].items():
            wandb.log({
                f'{threshold}_roc_auc': metrics['roc_auc'],
                f'{threshold}_avg_precision': metrics['avg_precision']
            })
    
    return results

def main_test():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    metadata_path = Path("datasets/Lav-DF/metadata.json")
    root_dir = Path("datasets/Lav-DF")
    


    test_loader = create_lav_df_dataloader(
        metadata_path=metadata_path,
        root_dir=root_dir,
        batch_size=1,
        frame_size=(112,112),
        goal_fps=1,
        
    )
    config = {
        "batch_size": 2,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dim_feedforward": 2048,
    }
    # Load model
    model = MultiModalFeatureExtractor(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"]
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoint_epoch_10.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Loss function and testing
    criterion = cross_modal_consistency_loss
    
    # Run comprehensive testing
    test_results = test_model_with_auc(
        model, 
        test_loader, 
        device, 
        criterion,
        thresholds=[6.7, 6.72,6.74, 6.76, 6.78, 6.8],
        use_wandb=False
    )
    
    # Print detailed results
    print_detailed_results(test_results)

if __name__ == "__main__":
    main_test()