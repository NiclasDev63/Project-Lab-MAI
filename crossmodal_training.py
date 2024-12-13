import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import whisper
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, read_video_timestamps

from AdaFace.face_alignment import align
from AdaFace.inference import load_pretrained_model, to_input
from data_loader.vox_celeb2.video_transforms import (
    NormalizeVideo,
)


class MultiModalFeatureExtractor(nn.Module):
    def __init__(
        self,
        d_model=1280,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=5120,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Memory tracking function
        def log_memory_usage(message=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024 / 1024  # Convert to GB
                reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024 / 1024  # Convert to GB
                print(f"{message} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
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

        self.log_memory_usage("After Image Alignment")

        frame_features = []
        frames_reshaped = torch.reshape(frames,[batch_size * num_frames, channels, height, width]).contiguous()
        
        features = self.adaface(frames_reshaped)[0] 
        frame_features = features.view(batch_size, num_frames, -1)

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
            feature_length = int((length / 30) * self.whisper.dims.n_audio_ctx)
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
        
        return visual_features, aligned_audio
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
        visual_features_out = []
        audio_features_out = []
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
            seq_visual_out, seq_audio_out = self.align_and_combine(
                seq_visual, seq_audio, seq_timestamps, audio_timestamps
            )
            visual_features_out.append(seq_visual_out)
            audio_features_out.append(seq_audio_out)
            # Periodically check memory
            if i % 5 == 0:
                self.log_memory_usage(f"During Combine Features - Sequence {i}")

        # Stack combined features
        visual_features_out = torch.stack(visual_features_out, dim=0)
        audio_features_out = torch.stack(audio_features_out, dim=0)
        self.log_memory_usage("End of Forward Pass")

        return visual_features_out, audio_features_out

class VoxCeleb2Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        frame_size=(224, 224),
        max_video_length=30,  # Fixed maximum video length in seconds
        max_audio_length=30,  # Maximum audio length in seconds for Whisper
        goal_fps=None,  # Optional frame rate reduction
        n_mels=128,
        train_list_path="datasets/test/train_list.txt",
        max_videos = 100000
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "dev" if split == "train" else "test"
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
                parts = line.strip().split()
                identity = parts[0]
                video_path = self.root_dir / "dev" / Path("mp4") / (parts[1][:-3] + "mp4")

                if identity not in self.videos_by_identity:
                    self.videos_by_identity[identity] = []
                    if len(self.video_paths) < max_videos:
                        self.video_paths.append(video_path)
                self.videos_by_identity[identity].append(video_path)


        for i in range(1,30):
           for videos in self.videos_by_identity.values():
               if len(self.video_paths) >= max_videos: 
                   break
               if len (videos) > i:
                   self.video_paths.append(videos[i])
            
            
            
        # Initialize Whisper processor for mel spectrograms
        self.whisper_processor = whisper.log_mel_spectrogram
        
        self.video_transforms = transforms.Compose(
            [
                # Use mean and std as described in AdaFace github repo
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _process_frame(self, frame):
        """Convert frame to PIL, align face, and transform to input tensor."""
        pil_image = Image.fromarray(frame.numpy().transpose(1, 2, 0).astype(np.uint8))  # Convert to PIL format
        aligned_rgb_img = align.get_aligned_face(image_path=None, rgb_pil_image=pil_image)
        bgr_input = to_input(aligned_rgb_img)
        bgr_input = bgr_input.squeeze(0)
        return bgr_input

    def _load_video_frames(self, video_path):
        """Load and process video frames with optional frame rate reduction."""
        pts, fps = read_video_timestamps(str(video_path))
        frames, audio, info = read_video(
            str(video_path),
            output_format="TCHW",  # Returns frames in format (time, channels, height, width)
        )

        # Optional frame rate reduction
        if self.goal_fps and self.goal_fps < info['video_fps']:
            skip_interval = max(1, int(round(info['video_fps'] / self.goal_fps)))
            frames = frames[::skip_interval]
            pts = pts[::skip_interval]  # Make sure to adjust the timestamps accordingly

        total_frames = int(self.max_video_length * (self.goal_fps or info['video_fps']))
        #trim 
        if frames.size(0) > total_frames:
            frames = frames[:total_frames]
            pts = pts[:total_frames]
        # Convert frames to PIL, process, and return tensor
        try:
            processed_frames = torch.stack([self._process_frame(frame) for frame in frames])
            processed_frames = self.video_transforms(processed_frames) 
            valid = True
        except Exception as e:
            processed_frames =  torch.zeros((len(frames),3,112,112))
            valid = False
        

        # Pad with black frames if needed
        padding_frames = total_frames - processed_frames.size(0)
        if padding_frames > 0:
            black_frame = torch.zeros_like(processed_frames[0])
            processed_frames = torch.cat(
                [processed_frames, black_frame.repeat(padding_frames, 1, 1, 1)],
                dim=0
            )

            # Pad pts (timestamps) to match the number of frames
            if len(pts) > 1:
                interval = pts[1] - pts[0]  # Calculate the interval between timestamps
                last_pt = pts[-1]  # Get the last timestamp
                additional_pts = [last_pt + interval * (i + 1) for i in range(padding_frames)]
                additional_pts_tensor = torch.tensor(additional_pts)

            # Concatenate the padded pts with the additional_pts
                pts = torch.cat([torch.tensor(pts), additional_pts_tensor])
        else:
            pts = torch.tensor(pts)

        return processed_frames, audio, info, pts, valid

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Load frames, audio, and metadata

        frames, audio, info, pts, valid = self._load_video_frames(video_path)




        audio_len =audio.size(1)/16000    
        audio = whisper.pad_or_trim(audio)
        mel = self.whisper_processor(audio.squeeze(1).numpy(), n_mels=self.n_mels).squeeze(0)
          # Use Whisper's method to pad or trim
        return {
            "frames": frames,  # shape: (total_frames, 3, H, W)
            "mel_spectrogram": mel,  # shape: (1, n_mels, T)
            "audio_length": audio_len,  # scalar
            "video_path": str(video_path),  # for debugging
            "frame_times": pts,  # Timestamp of the frames (padded)
            "valid": valid,
        }


def create_voxceleb2_dataloader(
    root_dir,
    batch_size=8,
    num_workers=0,
    split="train",
    frame_size=(112, 112),
    max_video_length=30,
    max_audio_length=30,
    goal_fps=None,
    n_mels=128,
    max_videos = 100000
):
    """
    Create a DataLoader for the VoxCeleb2 dataset
    """
    dataset = VoxCeleb2Dataset(
        root_dir=root_dir,
        split=split,
        frame_size=frame_size,
        max_video_length=max_video_length,
        max_audio_length=max_audio_length,
        goal_fps=goal_fps,
        n_mels=n_mels,
        max_videos = max_videos
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
