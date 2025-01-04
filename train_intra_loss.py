import argparse
import os
import uuid
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from AdaFace.inference import load_pretrained_model
from data_loader.vox_celeb2.VoxCeleb2Ada import create_voxceleb2_adaface_dataloader
from loss_function import IntraModalConsistencyLoss

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")

wandb.login(key=WANDB_API_KEY)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)


def _extract_identity_features(model, frames):
    """
    Extract features for all identities and their video frames efficiently.

    Args:
        model (nn.Module): The feature extractor model.
        frames (torch.Tensor): A batch of videos, shape (batch_size, num_frames, channels, height, width).

    Returns:
        torch.Tensor: Extracted features per identity, shape (batch_size, feature_dim).
    """
    batch_size, num_frames, channels, height, width = frames.shape

    # Reshape frames to (batch_size * num_frames, channels, height, width)
    frames = frames.view(batch_size * num_frames, channels, height, width)

    # Pass all frames through the model at once (parallel processing for each frame)
    frame_features = model(frames)[
        0
    ]  # Output shape: (batch_size * num_frames, feature_dim)

    # Reshape back to (batch_size, num_frames, feature_dim)
    identity_features = frame_features.view(batch_size, num_frames, -1)

    return identity_features


def train_epoch(model, train_loader, optimizer, device, criterion):
    """
    Train the model for one epoch

    Args:
        model (nn.Module): The multi-modal feature extractor
        train_loader (DataLoader): Training data loader
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
        # Zero the parameter gradients
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        ):
            # Move batch to device
            frames = batch["frames"]
            frames = frames.to(device)

            frame_features = _extract_identity_features(model, frames)

            loss = criterion(frame_features)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        mem = torch.cuda.memory_allocated(device)
        print("CURRENT MEMORY ALLOCATED: ", mem)

        # Update metrics
        total_loss += loss.item()
        batch_count += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0

    return {"loss": avg_loss}


def validate_model(model, val_loader, device, criterion):
    """
    Validate the model

    Args:
        model (nn.Module): The multi-modal feature extractor
        val_loader (DataLoader): Validation data loader
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

            frames = batch["frames"]
            frames = frames.to(device)

            frame_features = _extract_identity_features(model, frames)

            loss = criterion(frame_features)

            # Update metrics
            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"val_loss": loss.item()})

            mem = torch.cuda.memory_allocated(device)
            print("CURRENT MEMORY ALLOCATED: ", mem)

    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0

    return {"val_loss": avg_loss}


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_voxceleb2_adaface_dataloader(
        split="train",
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_loader = create_voxceleb2_adaface_dataloader(
        split="val",
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = load_pretrained_model("ir_50").to(device)
    model = torch.compile(model)
    criterion = IntraModalConsistencyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": criterion.parameters()},
        ],
        lr=args.learning_rate,
        weight_decay=2e-1,
    )

    experiment_name = "intra-modal-consistency-loss-v1"
    run_name = f"{experiment_name}-{uuid.uuid4().hex[:8]}"

    config = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    wandb.init(
        project=experiment_name,
        name=run_name,
        config=config,
    )

    print("Starting training...")

    training_history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(args.epochs):

        train_metrics = train_epoch(model, train_loader, optimizer, device, criterion)

        val_metrics = validate_model(model, val_loader, device, criterion)

        training_history["train_loss"].append(train_metrics["loss"])
        training_history["val_loss"].append(val_metrics["val_loss"])

        current_temperature = criterion.temperature.item()

        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
                "temperature": current_temperature,
            },
            checkpoint_path,
        )

        # Log metrics separately
        wandb.log(
            {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
                "temperature": current_temperature,
            }
        )

        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Temperature: {current_temperature:.4f}")

    print("Training completed!")


def set_start_method_spawn():
    """Safely set the start method to spawn if it hasn't been set yet."""
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn")
    except RuntimeError:
        pass


if __name__ == "__main__":
    set_start_method_spawn()
    main()
