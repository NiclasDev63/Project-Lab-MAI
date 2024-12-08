import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from AdaFace.inference import load_pretrained_model
from data_loader.vox_celeb2.VoxCeleb2Ada import (
    VoxCeleb2Ada,
    create_voxceleb2_adaface_dataloader,
)

# from crossmodal_training import VoxCeleb2Dataset, create_voxceleb2_dataloader
from loss_function import intra_modal_consistency_loss
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = VoxCeleb2Dataset("datasets", split="test")
# dataloader = create_voxceleb2_dataloader(
#     root_dir="datasets", split="test", batch_size=8
# )

dataloader = create_voxceleb2_adaface_dataloader(
    root_dir="datasets/train", batch_size=8
)


model = load_pretrained_model("ir_50").to(device)


def train_adaface(model):
    # use 1e-3 as lr for fine tuning, since authors used 1e-2 to pre train AdaFace
    learning_rate = 1e-3
    epochs = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    losses = []
    num_batches = len(dataloader.dataset) // dataloader.batch_sampler.batch_size
    progress_bar = tqdm(total=num_batches, desc="Processing")

    for epoch in range(epochs):
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        for batch in tqdm(dataloader, desc=f"Batch Progress", leave=False):
            progress_bar.update(1)

            optimizer.zero_grad()
            frames = batch["frames"].to(device)
            batch_size, num_frames = frames.shape[:2]
            frame_features = []

            for i in range(batch_size):
                batch_frames = frames[i].contiguous()
                identity_features = model(batch_frames)[0]
                frame_features.append(identity_features)
            frame_features = torch.stack(frame_features)

            # Compute loss
            loss = intra_modal_consistency_loss(frame_features)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()

            optimizer.step()

            # Adjust learning rate
            # scheduler.step()

        # Save final model
        torch.save(model.state_dict(), "adaface_finetuned.pth")


if __name__ == "__main__":
    train_adaface(model)
