import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import whisper
from AdaFace.inference import load_pretrained_model

class TemporalAlignmentModule(nn.Module):
    """
    Module to align and combine frame-level visual features with corresponding audio features
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, visual_features, audio_features, audio_timestamps, frame_timestamps):
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
        combined_features = torch.cat([visual_features, aligned_audio], dim=-1)  # Shape: (num_frames, 1792)
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize AdaFace for visual features
        self.adaface = load_pretrained_model("ir_50")
        
        # Initialize Whisper for audio features
        self.whisper = whisper.load_model("large-v3")
        self.audio_encoder = self.whisper.encoder
        del self.whisper.decoder
        
        # Transformer encoder for temporal aggregation of visual features
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.visual_transformer = TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
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
            features = self.adaface(frame)[0]  # Get identity features
            frame_features.append(features)
        
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, 512)
        
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
    
    def align_and_combine(self, visual_features, audio_features, frame_timestamps, audio_timestamps):
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
        combined_features = torch.cat([visual_features, aligned_audio], dim=-1)  # (num_frames, 1792)
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
            audio_timestamps = torch.linspace(0, original_lengths[i], audio_time, device=self.device)
            
            # Align and combine features
            seq_combined = self.align_and_combine(
                seq_visual,
                seq_audio,
                seq_timestamps,
                audio_timestamps
            )
            combined_features.append(seq_combined)
        
        # Stack combined features
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features



def train_multimodal_system(
    model,
    train_loader,
    num_epochs=10,
    learning_rate=1e-4,
    device='cuda'
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (video_frames, audio_data, frame_times) in enumerate(train_loader):
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
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')