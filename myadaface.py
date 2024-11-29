from pathlib import Path
import random
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

from AdaFace.inference import load_pretrained_model
import time
from tqdm import tqdm
from crossmodal_training import (
    MultiModalFeatureExtractor,
    VoxCeleb2Dataset,
    create_voxceleb2_dataloader,
)
from loss_function import intra_modal_consistency_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = VoxCeleb2Dataset("datasets", split="test")
dataloader = create_voxceleb2_dataloader(
    root_dir="datasets", split="test", batch_size=2, num_workers=2
)
extractor = MultiModalFeatureExtractor()
model = load_pretrained_model("ir_50").to(device)


def get_all_identities(split="test"):
    """
    Returns a list of all unique identity IDs in the dataset.
    """
    root_dir = Path(dataset.root_dir)
    split_dir = root_dir / ("dev" if split == "train" else "test")
    identities = [d.name for d in split_dir.iterdir() if d.is_dir()]
    return identities


def get_identity_videos(split="test"):
    """
    Returns a dictionary mapping each identity ID to a list of their video paths.
    """
    root_dir = Path(dataset.root_dir)
    split_dir = root_dir / ("dev" if split == "train" else "test")
    identity_videos = {}

    for person_dir in split_dir.iterdir():
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        video_paths = []

        for video_dir in person_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for video_file in video_dir.glob("*.mp4"):
                video_paths.append(video_file)

        if video_paths:
            identity_videos[person_id] = video_paths

    return identity_videos


def train_adaface(model):
    # Training parameters
    learning_rate = 3e-6

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    total_loss = 0

    for batch in dataloader:

        last_time = time.time()
        optimizer.zero_grad()
        frames = batch["frames"].to(device)
        batch_size, num_frames = frames.shape[:2]
        frame_features = []

        # Process each frame individually through AdaFace
        for i in range(num_frames):
            frame = frames[:, i]  # (batch_size, 3, 112, 112)
            frame = frame.contiguous()
            features = model(frame)[0]  # Get identity features
            frame_features.append(features)

        # Stack frame features
        frame_features = torch.stack(
            frame_features, dim=1
        )  # (batch_size, num_frames, 512)
        # Move features to device
        frame_features = frame_features.to(device)

        # Forward pass through the model (if applicable)
        # If the model processes the features further, include it here

        # Compute loss
        loss = intra_modal_consistency_loss(frame_features)

        current_time = time.time()
        print(f"Loss computation took {current_time - last_time:.4f} seconds")
        last_time = current_time

        # Backpropagation
        loss.backward()

        current_time = time.time()
        print(f"Backward pass took {current_time - last_time:.4f} seconds")
        last_time = current_time

        optimizer.step()

        current_time = time.time()
        print(f"Optimizer step took {current_time - last_time:.4f} seconds")
        last_time = current_time

        # Accumulate and display loss
        total_loss += loss.item()
        avg_loss = total_loss / (batch + 1)
        progress_bar.set_postfix({"loss": avg_loss})
        progress_bar.update(1)  # Advance the progress bar

        # Adjust learning rate
        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), "adaface_finetuned.pth")


if __name__ == "__main__":

    train_adaface(model)
