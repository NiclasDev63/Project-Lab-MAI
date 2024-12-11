import os
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms
import whisper
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, read_video_timestamps
from torchvision.transforms.functional import InterpolationMode

from AdaFace.face_alignment import align
from AdaFace.inference import load_pretrained_model, to_input
from data_loader.vox_celeb2.video_transforms import (
    NormalizeVideo,
    ResizeVideo,
    SquareVideo,
    ToTensorVideo,
)


class TemporalAlignmentModule(nn.Module):
    """
    Module to align and combine frame-level visual features with corresponding audio features
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, visual_features, audio_features, audio_timestamps, frame_timestamps
    ):
        """
        Aligns visual and audio features based on timestamps and concatenates them

        Args:
            visual_features: tensor of shape (num_frames, 512) - from AdaFace
            audio_features: tensor of shape (audio_time, 1280) - from Whisper
            audio_timestamps: tensor of shape (audio_time,) in seconds
            frame_timestamps: tensor of shape (num_frames,) in seconds

        Returns:
            combined_features: tensor of shape (num_frames, 1792)  # 512 + 1280
        """
        # For each frame timestamp, find the closest audio timestamp
        frame_indices = []
        for frame_time in frame_timestamps:
            # Find closest audio timestamp
            distances = torch.abs(audio_timestamps - frame_time)
            closest_idx = torch.argmin(distances)
            frame_indices.append(closest_idx)

        # Get corresponding audio features
        aligned_audio = audio_features[frame_indices]

        # Simply concatenate visual and aligned audio features
        combined_features = torch.cat(
            [visual_features, aligned_audio], dim=-1
        )  # Shape: (num_frames, 1792)
        return combined_features


class MultiModalFeatureExtractor(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Memory tracking function
        def log_memory_usage(message=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # Convert to MB
                reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024  # Convert to MB
                print(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
                torch.cuda.reset_peak_memory_stats(self.device)

        # Log initial memory
        log_memory_usage("Initial Memory")

        # Initialize AdaFace for visual features
        adaface = load_pretrained_model("ir_50")
        self.adaface = adaface.to(self.device)
        self.adaface.train()
        log_memory_usage("After AdaFace Loading")

        # Initialize Whisper for audio features
        self.whisper = whisper.load_model("turbo").to(self.device)
        self.audio_encoder = self.whisper.encoder
        del self.whisper.decoder
        log_memory_usage("After Whisper Loading")

        # Transformer encoder for temporal aggregation of visual features
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.visual_transformer = TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        ).to(self.device)
        log_memory_usage("After Transformer Loading")

        # Store memory logging function
        self.log_memory_usage = log_memory_usage

    def process_frames(self, frames, original_lengths):
        """
        Process individual frames through AdaFace and transformer
        """
        self.log_memory_usage("Before Frame Processing")

        batch_size, num_frames, channels, height, width = frames.shape
        frames_reshaped = frames.view(batch_size * num_frames, channels, height, width)
        
        # Manage memory for image processing
        pil_images = [Image.fromarray(frame.permute(1, 2, 0).byte().cpu().numpy()) for frame in frames_reshaped]
        del frames_reshaped  # Free up memory
        
        aligned_pil_images = [align.get_aligned_face("", rgb_pil_image=img) for img in pil_images]
        del pil_images  # Free up memory
        
        aligned_frames = torch.stack([to_input(img) for img in aligned_pil_images])
        del aligned_pil_images  # Free up memory
        
        aligned_frames = aligned_frames.view(batch_size, num_frames, *aligned_frames.shape[1:])
        self.log_memory_usage("After Image Alignment")

        frame_features = []
        
        # Process each frame individually through AdaFace
        for i in range(num_frames):
            self.adaface.train()
            frame = frames[:, i]
            frame = frame.contiguous()
            features = self.adaface(frame)[0]  # Get identity features
            
            frame_features.append(features)
            
            # Periodically check memory
            if i % 10 == 0:
                self.log_memory_usage(f"During Frame Processing - Frame {i}")

        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)
        self.log_memory_usage("After Frame Feature Extraction")

        # Process through transformer
        transformed_features = self.visual_transformer(frame_features)
        self.log_memory_usage("After Visual Transformer")

        return transformed_features

    def process_audio(self, mel_features, original_lengths):
        """
        Process audio through Whisper encoder and extract relevant features
        """
        self.log_memory_usage("Before Audio Processing")

        # Process through Whisper encoder
        audio_features = self.audio_encoder(mel_features)
        self.log_memory_usage("After Whisper Audio Encoding")

        # Extract only the relevant features based on original lengths
        extracted_features = []
        for features, length in zip(audio_features, original_lengths):
            # Convert audio length to feature length
            feature_length = length // self.whisper.dims.n_audio_ctx
            # Extract only the valid features
            valid_features = features[:feature_length]
            extracted_features.append(valid_features)

        self.log_memory_usage("After Audio Feature Extraction")
        return extracted_features
    def align_and_combine(
        self, visual_features, audio_features, frame_timestamps, audio_timestamps
    ):
        """
        Align and combine visual and audio features

        Args:
            visual_features: tensor of shape (num_frames, 512)
            audio_features: tensor of shape (audio_time, 1280)
            frame_timestamps: tensor of shape (num_frames,)
            audio_timestamps: tensor of shape (audio_time,)
        """
        # For each frame timestamp, find the closest audio timestamp
        frame_indices = []
        for frame_time in frame_timestamps:
            distances = torch.abs(audio_timestamps - frame_time)
            closest_idx = torch.argmin(distances).item()
            frame_indices.append(closest_idx)


        # Get corresponding audio features and concatenate
        aligned_audio = audio_features[frame_indices]
        combined_features = torch.cat(
            [visual_features, aligned_audio], dim=-1
        )  # (num_frames, 1792)
        return combined_features
    def forward(self, frames, mel_features, original_lengths, frame_timestamps):
        """
        Forward pass processing full audio and individual frames
        """
        self.log_memory_usage("Start of Forward Pass")

        batch_size = frames.shape[0]

        # Process all frames through AdaFace and transformer
        visual_features = self.process_frames(frames, original_lengths)
        self.log_memory_usage("After Visual Feature Processing")

        # Process full audio through Whisper and get relevant features
        audio_features = self.process_audio(mel_features, original_lengths)
        self.log_memory_usage("After Audio Feature Processing")

        # Align and combine features for each sequence in the batch
        combined_features = []
        for i in range(batch_size):
            # Get features for current sequence
            seq_visual = visual_features[i]
            seq_audio = audio_features[i]
            seq_timestamps = frame_timestamps[i]

            # Generate audio timestamps based on actual feature length
            audio_time = seq_audio.shape[0]
            audio_timestamps = torch.linspace(
                0, original_lengths[i], audio_time, device=self.device
            )

            # Align and combine features
            seq_combined = self.align_and_combine(
                seq_visual, seq_audio, seq_timestamps, audio_timestamps
            )
            combined_features.append(seq_combined)

            # Periodically check memory
            if i % 5 == 0:
                self.log_memory_usage(f"During Combine Features - Sequence {i}")

        # Stack combined features
        combined_features = torch.stack(combined_features, dim=0)
        self.log_memory_usage("End of Forward Pass")

        return combined_features

class VoxCeleb2Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        frames_per_clip=25,
        frame_size=(112, 112),
        max_video_length=10,  # Maximum video length in seconds
        max_audio_length=30,  # Maximum audio length in seconds for Whisper
        goal_fps=None,  # Optional frame rate reduction
        n_mels=128,
        train_list_path="datasets/test/train_list.txt"
    ):
        """
        Args:
            root_dir: Path to VoxCeleb2 dataset root
            split: 'train' or 'test'
            frames_per_clip: Maximum number of frames to sample
            frame_size: Size to resize frames to (height, width)
            max_video_length: Maximum video length in seconds for frames
            max_audio_length: Maximum audio length in seconds (for Whisper)
            goal_fps: Optional target frame rate to reduce video frames
            n_mels: Number of mel spectrogram bins
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "dev" if split == "train" else "test"
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.max_video_length = max_video_length
        self.max_audio_length = max_audio_length
        self.goal_fps = goal_fps
        self.n_mels = n_mels

        # Load video paths
        
        self.video_paths = []
        self.videos_by_identity = {}
        with open(train_list_path, 'r') as f:
            for line in f:
                # Assuming the format is: identity video_path
                parts = line.strip().split()
                identity = parts[0]
                video_path = self.root_dir / "dev" / Path("mp4") / (parts[1][:-3] + "mp4")
                
                if identity not in self.videos_by_identity:
                    self.videos_by_identity[identity] = []
                self.videos_by_identity[identity].append(video_path)
                self.video_paths.append(video_path)
        
        # self.video_paths = []
        # split_dir = self.root_dir / self.split
        
        # for person_id in os.listdir(split_dir):
        #     person_dir = split_dir / person_id
        #     if not person_dir.is_dir():
        #         continue

        #     for video_id in os.listdir(person_dir):
        #         video_dir = person_dir / video_id
        #         if not video_dir.is_dir():
        #             continue

        #         for video_file in video_dir.glob("*.mp4"):
        #             self.video_paths.append(video_file)

        # Video transforms from marcels file
        self.video_transforms = transforms.Compose(
            [
                SquareVideo(),
                # removed due to frame processing needing rgb convert rgb to bgr as described in AdaFace github repo see: https://github.com/mk-minchul/AdaFace?tab=readme-ov-file#general-inference-guideline
                #transforms.Lambda(lambda x: x[:, [2, 1, 0], :, :]),
                ResizeVideo(frame_size, InterpolationMode.BILINEAR),
                ToTensorVideo(max_pixel_value=255.0),
                # use mean and std as describe in AdaFace github repo see: https://github.com/mk-minchul/AdaFace?tab=readme-ov-file#general-inference-guideline
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Initialize Whisper processor for mel spectrograms
        self.whisper_processor = whisper.log_mel_spectrogram

    def _load_video_frames(self, video_path):
        """Load and process video frames with optional frame rate reduction"""
        # Get video timestamps first
        pts, fps = read_video_timestamps(str(video_path))
        
        # Convert pts to tensor
        pts_tensor = torch.tensor(pts)

        # Read video 
        frames, audio, info = read_video(
            str(video_path),
            output_format="TCHW",  # Returns frames in format (time, channels, height, width)
        )

        # Optional frame rate reduction
        if self.goal_fps and self.goal_fps < info['video_fps']:
            # Calculate frame skip interval
            skip_interval = max(1, int(round(info['video_fps'] / self.goal_fps)))
            frames = frames[::skip_interval]
            pts_tensor = pts_tensor[::skip_interval]

        # Trim or pad frames to max_video_length
        max_frames = int(self.max_video_length * info['video_fps'])
        if frames.size(0) > max_frames:
            frames = frames[:max_frames]
            pts_tensor = pts_tensor[:max_frames]
        
        # Apply video transforms
        frames = self.video_transforms(frames)
        frame_times = pts_tensor
        
        return frames, frame_times, audio, info

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Load frames, audio, and metadata
        frames, frame_times, audio, info = self._load_video_frames(video_path)

        # Process audio
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=1, keepdim=True)

        # Resample to 16kHz if needed (Whisper's expected sample rate)
        if info["audio_fps"] != 16000:
            resampler = torchaudio.transforms.Resample(info["audio_fps"], 16000)
            audio = resampler(audio.t()).t()

        # Convert to mel spectrogram and pad (or trim) using Whisper's processor
        mel = self.whisper_processor(
            audio.squeeze(1).numpy(), n_mels=self.n_mels
        ).squeeze(0)

        return {
            "frames": frames,  # shape: (frames_per_clip, 3, H, W)
            "mel_spectrogram": mel,  # shape: (1, n_mels, T)
            "audio_length": audio.size(1),  # scalar
            "frame_times": frame_times,  # shape: (frames_per_clip,)
            "video_path": str(video_path),  # for debugging
        }


def create_voxceleb2_dataloader(
    root_dir,
    batch_size=8,
    num_workers=4,
    split="train",
    frames_per_clip=16,
    frame_size=(112, 112),
    max_video_length=10,
    max_audio_length=30,
    goal_fps=None,
    n_mels=128,
):
    """
    Create a DataLoader for the VoxCeleb2 dataset
    """
    dataset = VoxCeleb2Dataset(
        root_dir=root_dir,
        split=split,
        frames_per_clip=frames_per_clip,
        frame_size=frame_size,
        max_video_length=max_video_length,
        max_audio_length=max_audio_length,
        goal_fps=goal_fps,
        n_mels=n_mels,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    return dataloader


def custom_collate_fn(batch):
    """
    Custom collate function to pad:
    - mel spectrograms to a fixed length (3000)
    - frames and frame_times with -1 for variable length sequences
    Custom collate function to pad:
    - mel spectrograms to a fixed length (3000)
    - frames and frame_times with -1 for variable length sequences
    """
    FIXED_LENGTH = 3000

    # Separate each item in the batch
    frames = [item["frames"] for item in batch]
    mel_spectrograms = [torch.tensor(item["mel_spectrogram"]) for item in batch]
    audio_lengths = [item["audio_length"] for item in batch]
    frame_times = [item["frame_times"] for item in batch]
    video_paths = [item["video_path"] for item in batch]

    # Find maximum sequence length for frames
    max_frames_length = max(f.size(0) for f in frames)
    
    # Pad frames with -1
    padded_frames = []
    for frame_seq in frames:
        padding_length = max_frames_length - frame_seq.size(0)
        if padding_length > 0:
            # Create padding tensor with same spatial dimensions as frames
            padding = torch.full(
                (padding_length, frame_seq.size(1), frame_seq.size(2), frame_seq.size(3)),
                -1.0,
                dtype=frame_seq.dtype
            )
            padded_frames.append(torch.cat([frame_seq, padding], dim=0))
        else:
            padded_frames.append(frame_seq)
    
    # Pad frame_times with -1
    padded_frame_times = []
    for time_seq in frame_times:
        padding_length = max_frames_length - time_seq.size(0)
        if padding_length > 0:
            padding = torch.full((padding_length,), -1.0, dtype=time_seq.dtype)
            padded_frame_times.append(torch.cat([time_seq, padding], dim=0))
        else:
            padded_frame_times.append(time_seq)

    # Stack all tensors
    frames = torch.stack(padded_frames)
    frame_times = torch.stack(padded_frame_times)
    audio_lengths = torch.tensor(audio_lengths)
    
    # Pad mel spectrograms to the fixed length (3000)
    padded_mels = [torch.nn.functional.pad(mel, (0, FIXED_LENGTH - mel.shape[-1])) for mel in mel_spectrograms]
    mel_spectrograms = torch.stack(padded_mels)

    return {
        "frames": frames,  # shape: (batch_size, max_seq_length, C, H, W)
        "mel_spectrogram": mel_spectrograms,
        "audio_length": audio_lengths,
        "frame_times": frame_times,  # shape: (batch_size, max_seq_length)
        "video_path": video_paths,
    }

