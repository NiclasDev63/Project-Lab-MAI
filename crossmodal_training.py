import random
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import whisper
from AdaFace.inference import load_pretrained_model
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video, read_video_timestamps
import os
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from data_loader.vox_celeb2.video_transforms import (
    SquareVideo,
    ResizeVideo,
    ToTensorVideo,
    NormalizeVideo,
)
import torchaudio
from torch.utils.data.sampler import Sampler


class IdentityBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.identities = dataset.identities

    def __iter__(self):
        # Shuffle identities for each epoch
        shuffled_identities = random.sample(self.identities, len(self.identities))
        for i in range(0, len(shuffled_identities), self.batch_size):
            yield [
                self.dataset.identities.index(identity)
                for identity in shuffled_identities[i : i + self.batch_size]
            ]

    def __len__(self):
        return len(self.identities) // self.batch_size


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

        # Initialize AdaFace for visual features
        self.adaface = load_pretrained_model("ir_50")

        # Initialize Whisper for audio features
        self.whisper = whisper.load_model("tiny")
        self.audio_encoder = self.whisper.encoder
        del self.whisper.decoder

        # Transformer encoder for temporal aggregation of visual features
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.visual_transformer = TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

    def process_frames(self, frames):
        """
        Process individual frames through AdaFace and transformer

        Args:
            frames: tensor of shape (batch_size, num_frames, 3, 112, 112)
        """
        batch_size, num_frames = frames.shape[:2]
        frame_features = []
        # Process each frame individually through AdaFace
        for i in range(num_frames):
            frame = frames[:, i]  # (batch_size, 3, 112, 112)
            print(frame.shape)
            print(frames.shape)
            features = self.adaface(frame)[0]  # Get identity features
            frame_features.append(features)
        # Stack frame features
        frame_features = torch.stack(
            frame_features, dim=1
        )  # (batch_size, num_frames, 512)
        # Process through transformer
        transformed_features = self.visual_transformer(frame_features)
        return transformed_features

    def process_audio(self, mel_features, original_lengths):
        """
        Process audio through Whisper encoder and extract relevant features
        Args:
            mel_features: tensor of shape (batch_size, n_mels, time) - already padded using whisper.pad_or_trim
            original_lengths: tensor of shape (batch_size,) containing original audio lengths before padding
        """
        # Process through Whisper encoder
        audio_features = self.audio_encoder(mel_features)  # (batch_size, time, 1280)
        # Extract only the relevant features based on original lengths
        extracted_features = []
        for features, length in zip(audio_features, original_lengths):
            # Convert audio length to feature length (accounting for any downsampling in Whisper)
            feature_length = length // self.whisper.dims.n_audio_ctx
            # Extract only the valid features
            valid_features = features[:feature_length]
            extracted_features.append(valid_features)
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
            closest_idx = torch.argmin(distances)
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

        Args:
            frames: tensor of shape (batch_size, num_frames, 3, 112, 112)
            mel_features: tensor of shape (batch_size, n_mels, time) - already padded using whisper.pad_or_trim
            original_lengths: tensor of shape (batch_size,) containing original audio lengths
            frame_timestamps: tensor of shape (batch_size, num_frames) containing frame timestamps
        """
        batch_size = frames.shape[0]

        # Process all frames through AdaFace and transformer
        visual_features = self.process_frames(frames)

        # Process full audio through Whisper and get relevant features
        audio_features = self.process_audio(mel_features, original_lengths)

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

        # Stack combined features
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features


def train_multimodal_system(
    model, train_loader, num_epochs=10, learning_rate=1e-4, device="cuda"
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (video_frames, audio_data, frame_times) in enumerate(
            train_loader
        ):
            # Prepare batch data
            frames, audio, audio_lengths, timestamps = prepare_batch_data(
                video_frames, audio_data, frame_times
            )

            # c-c-c-cuda
            frames = frames.to(device)
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)
            timestamps = timestamps.to(device)

            # Forward pass
            combined_features = model(frames, audio, audio_lengths, timestamps)

            # TODO: Add loss here
            loss = compute_loss(combined_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")


class VoxCeleb2Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        frames_per_clip=16,
        frame_size=(112, 112),
        max_audio_length=30,  # maximum audio length in seconds
    ):
        """
        Args:
            root_dir: Path to VoxCeleb2 dataset root
            split: 'train' or 'test'
            frames_per_clip: Number of frames to sample from each video
            frame_size: Size to resize frames to (height, width)
            max_audio_length: Maximum audio length in seconds
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "dev" if split == "train" else "test"
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.max_audio_length = max_audio_length

        # Group videos by identity
        self.identity_videos = {}  # {identity_id: [video_path1, video_path2, ...]}
        split_dir = self.root_dir / self.split

        for person_id in os.listdir(split_dir):
            person_dir = split_dir / person_id
            if not person_dir.is_dir():
                continue

            self.identity_videos[person_id] = []
            for video_id in os.listdir(person_dir):
                video_dir = person_dir / video_id
                if not video_dir.is_dir():
                    continue

                for video_file in video_dir.glob("*.mp4"):
                    self.identity_videos[person_id].append(video_file)

        self.identities = list(self.identity_videos.keys())  # List of identities

        # Video transforms from marcels file
        self.video_transforms = transforms.Compose(
            [
                SquareVideo(),
                ResizeVideo(frame_size, InterpolationMode.BILINEAR),
                ToTensorVideo(max_pixel_value=255.0),
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize Whisper processor for mel spectrograms
        self.whisper_processor = whisper.log_mel_spectrogram

    def _load_video_frames(self, video_path):
        """Load and process video frames"""
        # Get video timestamps first
        pts, fps = read_video_timestamps(str(video_path))

        # Calculate frame indices to sample
        if len(pts) <= self.frames_per_clip:
            indices = torch.linspace(0, len(pts) - 1, self.frames_per_clip).long()
        else:
            indices = torch.linspace(0, len(pts) - 1, self.frames_per_clip).long()

        # Convert pts to tensor
        pts_tensor = torch.tensor(pts)

        # start_pts = min(pts)
        # end_pts = max(pts)

        # Read video at selected timestamps
        frames, audio, info = read_video(
            str(video_path),
            # start_pts=start_pts,
            # end_pts=end_pts,
            output_format="TCHW",  # Returns frames in format (time, channels, height, width)
        )
        frames = frames[indices]
        # Apply video transforms
        frames = self.video_transforms(frames)
        frame_times = pts_tensor[indices] / info["video_fps"]

        return frames, frame_times, audio, info

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, idx):
        # Select an identity
        identity = self.identities[idx]

        # Randomly choose one video for this identity
        video_path = random.choice(self.identity_videos[identity])

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

        # Trim to max_audio_length if necessary
        max_samples = int(self.max_audio_length * 16000)
        if audio.size(1) > max_samples:
            audio = audio[:max_samples]

        # Convert to mel spectrogram using Whisper's processor
        mel = self.whisper_processor(audio.squeeze(1).numpy())

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
    max_audio_length=30,
):
    """
    Create a DataLoader for the VoxCeleb2 dataset
    """
    dataset = VoxCeleb2Dataset(
        root_dir=root_dir,
        split=split,
        frames_per_clip=frames_per_clip,
        frame_size=frame_size,
        max_audio_length=max_audio_length,
    )
    batch_sampler = IdentityBatchSampler(dataset, batch_size)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    return dataloader


def custom_collate_fn(batch):
    """
    Custom collate function to pad mel spectrograms to a fixed length (480000).
    """
    FIXED_LENGTH = 480000

    # Separate each item in the batch
    frames = [item["frames"] for item in batch]
    mel_spectrograms = [torch.tensor(item["mel_spectrogram"]) for item in batch]
    audio_lengths = [item["audio_length"] for item in batch]
    frame_times = [item["frame_times"] for item in batch]
    video_paths = [item["video_path"] for item in batch]

    # Stack frames and frame_times as they are already the same size
    frames = torch.stack(frames)
    frame_times = torch.stack(frame_times)
    audio_lengths = torch.tensor(audio_lengths)

    # Pad mel spectrograms to the fixed length (480000)
    padded_mels = [
        torch.nn.functional.pad(mel, (0, FIXED_LENGTH - mel.shape[-1]))
        for mel in mel_spectrograms
    ]
    mel_spectrograms = torch.stack(padded_mels)

    return {
        "frames": frames,
        "mel_spectrogram": mel_spectrograms,
        "audio_length": audio_lengths,
        "frame_times": frame_times,
        "video_path": video_paths,
    }
