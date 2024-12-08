import argparse
import uuid

import matplotlib.pyplot as plt
import torch
import wandb
from AdaFace.inference import load_pretrained_model
from data_loader.vox_celeb2.VoxCeleb2Ada import create_voxceleb2_adaface_dataloader
from loss_function import intra_modal_consistency_loss
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-3)


def _extract_features_for_identity(model, frames):
    batch_size = frames.shape[0]
    frame_features = []
    for i in range(batch_size):
        batch_frames = frames[i].contiguous()
        identity_features = model(batch_frames)[0]

        frame_features.append(identity_features)
    return torch.stack(frame_features)


def train_epoch(model, train_loader, optimizer, device):
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
        # Move batch to device
        frames = batch["frames"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        frame_features = _extract_features_for_identity(model, frames)

        loss = intra_modal_consistency_loss(frame_features)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        batch_count += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Compute average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0

    return {"loss": avg_loss}


def validate_model(model, val_loader, device):
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
            # Move batch to device
            frames = batch["frames"].to(device)

            frame_features = _extract_features_for_identity(model, frames)

            loss = intra_modal_consistency_loss(frame_features)

            # Update metrics
            total_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"val_loss": loss.item()})

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    experiment_name = "intra-modal-consistency-loss-v1"
    run_name = f"{experiment_name}-{uuid.uuid4().hex[:8]}"

    config = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    # wandb.init(
    #     project=experiment_name,
    #     name=run_name,
    #     config=config,
    # )

    print("Starting training...")

    training_history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(args.epochs):

        train_metrics = train_epoch(model, train_loader, optimizer, device)

        val_metrics = validate_model(model, val_loader, device)

        training_history["train_loss"].append(train_metrics["loss"])
        training_history["val_loss"].append(val_metrics["val_loss"])

        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
            },
            checkpoint_path,
        )

        # Create and log artifact separately
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch-{epoch+1}",
            type="model",
            metadata={
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
            },
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        # Log metrics separately
        wandb.log(
            {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
            }
        )

        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(training_history["train_loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss_plot.png")
    wandb.log({"training_loss_plot": wandb.Image("training_loss_plot.png")})

    wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()
