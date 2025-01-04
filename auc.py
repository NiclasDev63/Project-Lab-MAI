import multiprocessing
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from data_loader.fake_av_celeb.main import create_fake_av_celeb_dataloader
from AdaFace.inference import load_pretrained_model


FAKE_SET_PATH = "../../data/FakeAVCeleb/FakeVideo-RealAudio"
REAL_SET_PATH = "../../data/FakeAVCeleb/RealVideo-RealAudio"
MODEL_PATH = "checkpoint_epoch2"


# Function to compute similarity matrix
def compute_similarity_matrix(model, video_frames, device):
    video_frames = video_frames.to(device)
    window_size = 5
    num_windows = video_frames.size(0) // window_size
    video_windows = (
        video_frames[: num_windows * window_size]
        .view(num_windows, window_size, *video_frames.shape[1:])
        .to(device)
    )

    identity_features = []
    for window in video_windows:
        with torch.no_grad():
            frame_features, _ = model(window)
            window_feature = frame_features.mean(dim=0).to(device)
            identity_features.append(window_feature)

    identity_features = torch.stack(identity_features).to(device)
    identity_features = F.normalize(identity_features, p=2, dim=-1)
    similarity_matrix = torch.mm(identity_features, identity_features.T)

    return similarity_matrix


# Function to process videos and compute scores
def process_videos(model, dataloader, device):
    scores = []
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)

        for video_idx in range(frames.size(0)):
            video_frames = frames[video_idx]
            similarity_matrix = compute_similarity_matrix(model, video_frames, device)
            intra_modal_score = torch.quantile(similarity_matrix.flatten(), 0.2).item()
            scores.append(intra_modal_score)

        print(f"Processed Batch {batch_idx + 1} / {total_batches}")

    return torch.tensor(scores, device=device)


# Function to calculate AUC and AP
def calculate_auc_and_ap(scores, labels):
    scores = scores.cpu()
    labels = labels.cpu()
    auc = roc_auc_score(labels.numpy(), scores.numpy())
    ap = average_precision_score(labels.numpy(), scores.numpy())
    return auc, ap


# Main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model
    model = load_pretrained_model(MODEL_PATH).to(device)

    # Initialize scores and labels
    all_scores = []
    all_labels = []

    # Directories and labels
    directories = [
        (FAKE_SET_PATH, 0),  # Fake videos
        (REAL_SET_PATH, 1),  # Real videos
    ]

    for root_dir, label in directories:
        print(f"Processing videos from: {root_dir}")

        # Create dataloader for the current directory
        dataloader = create_fake_av_celeb_dataloader(root_dir=root_dir)

        # Process videos and compute scores
        scores = process_videos(model, dataloader, device)

        # Create labels tensor
        labels = torch.full((len(scores),), label, device=device, dtype=torch.int64)

        # Append to all scores and labels
        all_scores.append(scores)
        all_labels.append(labels)

    # Concatenate all scores and labels
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate AUC and AP
    auc, ap = calculate_auc_and_ap(all_scores, all_labels)
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
