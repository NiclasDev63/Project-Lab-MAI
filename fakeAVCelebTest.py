import numpy as np
import torch.nn.functional as F
import torch
from data_loader.fake_av_celeb.main import create_fake_av_celeb_dataloader
from sklearn.metrics import roc_auc_score, average_precision_score
from AdaFace.inference import load_pretrained_model


# Function to compute similarity matrix
def compute_similarity_matrix(model, video_frames, temperature=0.7, device="cuda"):
    """
    Computes the intra-modal similarity matrix for the video frames.

    Args:
        frames (torch.Tensor): Shape (num_frames, channels, height, width).
        device (str): "cuda" or "cpu" for tensor operations.

    Returns:
        torch.Tensor: Similarity matrix (num_windows x num_windows).
    """
    video_frames = video_frames.to(device)  # Move frames to GPU/CPU
    window_size = 5
    num_windows = video_frames.size(0) // window_size
    video_windows = (
        video_frames[: num_windows * window_size]
        .view(num_windows, window_size, *video_frames.shape[1:])
        .to(device)
    )  # Shape: (num_windows, window_size, channels, height, width)

    # Process each window
    identity_features = []
    for window in video_windows:
        with torch.no_grad():
            frame_features, _ = model(window)  # Shape: [num_frames, feature_dim]
            window_feature = frame_features.mean(dim=0).to(device)  # Aggregate features
            identity_features.append(window_feature)

    # Stack aggregated features into tensor of shape [num_windows, feature_dim]
    identity_features = torch.stack(identity_features).to(device)

    # Normalize the identity features
    identity_features = F.normalize(identity_features, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.mm(identity_features, identity_features.T)

    return similarity_matrix.cpu()  # Return to CPU if needed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize dataloader and model
    dataloader = create_fake_av_celeb_dataloader(
        root_dir="datasets/FakeAVCeleb/FakeVideo-RealAudio"
    )
    model = load_pretrained_model("ir_50").to(device)  # Move model to GPU/CPU

    scores = []
    labels = [0] * len(dataloader.dataset)  # All videos labeled as fake (0)
    total_batches = len(dataloader)  # Total number of batches
    processed_batches = 0  # Counter for processed batches

    for batch in dataloader:
        frames = batch[
            "frames"
        ]  # Shape: (batch_size, num_frames, channels, height, width)
        frames = frames.to(device)  # Move frames to GPU/CPU

        # Process each video in the batch
        for id in range(frames.size(0)):
            video_frames = frames[id]  # Shape: (num_frames, channels, height, width)

            # Compute similarity matrix for the video
            similarity_matrix = compute_similarity_matrix(
                model, video_frames, device=device
            )

            # Calculate the 20th percentile of the similarity matrix
            intra_modal_score = np.percentile(similarity_matrix.numpy(), 20)
            scores.append(intra_modal_score)
            print("processing video ", id, " in Batch ", processed_batches)

        processed_batches += 1
        print(f"Processed Batch {processed_batches} / {total_batches}")
    np.savetxt("scores.txt", scores)
    # Compute AUC and AP
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    print("AUC:", auc)
    print("AP:", ap)


if __name__ == "__main__":
    main()
