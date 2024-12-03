### Test py to simulate a full training loop with less data for testing and debugging
from pathlib import Path

import torch
import whisper

from crossmodal_training import MultiModalFeatureExtractor, create_voxceleb2_dataloader

whisper.transcribe

feature_extractor = MultiModalFeatureExtractor()
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_dir = "datasets"
dataloader = create_voxceleb2_dataloader(
    Path(dataset_dir),
    split="test",
    num_workers=0,
    batch_size=1,
    n_mels=feature_extractor.whisper.dims.n_mels,
)


for batch in dataloader:
    frames = batch["frames"].to(device)
    audio = batch["mel_spectrogram"].squeeze(1).to(device)
    audio_lengths = batch["audio_length"].to(device)
    timestamps = batch["frame_times"].to(device)

    print("AUDIO PATH: ", batch["video_path"])

    features = feature_extractor(frames, audio, audio_lengths, timestamps)
    print(features.shape)
    break
