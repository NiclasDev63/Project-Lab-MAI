import multiprocessing as mp
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from AdaFace.face_alignment import align
from AdaFace.inference import load_pretrained_model, to_input
from data_loader.fake_av_celeb.main import create_fake_av_celeb_dataloader
from data_loader.vox_celeb2.VoxCeleb2Ada import create_voxceleb2_adaface_dataloader
from loss_function import IntraModalConsistencyLoss


def set_start_method_spawn():
    """Safely set the start method to spawn if it hasn't been set yet."""
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn")
    except RuntimeError:
        pass


def _extract_identity_features(model, frames):
    """Extract features for all identities and their video frames efficiently."""
    batch_size, num_frames, channels, height, width = frames.shape
    frames = frames.view(batch_size * num_frames, channels, height, width)
    frame_features, norms = model(frames)

    # Average the features across frames for each identity
    identity_features = frame_features.view(batch_size, num_frames, -1)
    identity_features = torch.mean(
        identity_features, dim=1
    )  # (batch_size, feature_dim)

    return identity_features


def validate_model(model, val_loader, device):
    """Validate the model using AUC metric for fake detection"""
    model.eval()
    features_by_identity = defaultdict(list)

    # First, collect all features grouped by identity
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting features"):
            frames = batch["frames"].to(device)
            identity_idx = batch["speaker_id"]

            features = _extract_identity_features(model, frames)

            # Group features by identity
            for feat, idx in zip(features, identity_idx):
                features_by_identity[idx].append(feat)

    all_similarities = []
    all_labels = []

    # For each identity, compute similarities between all its videos
    for identity in features_by_identity:
        if len(features_by_identity[identity]) > 1:  # Only if we have multiple videos
            features = torch.stack(features_by_identity[identity])

            # Compute similarity matrix for this identity's videos
            similarity_matrix = torch.mm(features, features.t())

            # Get upper triangle of matrix (excluding diagonal)
            mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1)
            similarities = similarity_matrix[mask.bool()]

            # All comparisons are between fake videos of same identity
            labels = torch.zeros_like(similarities)

            all_similarities.append(similarities)
            all_labels.append(labels)

    # Concatenate all similarities and labels
    if all_similarities:  # Check if we have any comparisons
        final_similarities = torch.cat(all_similarities)
        final_labels = torch.cat(all_labels)

    else:
        auc_score = torch.tensor(0.0)

    return {"auc": auc_score.item()}


def main():

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    assert device_name == "cuda", "No cuda available"
    device = torch.device(device_name)
    model = load_pretrained_model("ir_50")
    model = model.to(device)
    # criterion = IntraModalConsistencyLoss()
    # criterion_state_dict = torch.load("./AdaFace/pretrained/checkpoint_epoch_1.pth")[
    #     "criterion_state_dict"
    # ]
    # criterion.load_state_dict(criterion_state_dict)

    # data_root = "../../data/vox2"
    data_root = "../FakeAVCeleb/FakeVideo-RealAudio"
    batch_size = 8
    num_workers = 4
    # val_loader = create_voxceleb2_adaface_dataloader(
    #     split="val",
    #     root_dir=data_root,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    # )
    val_loader = create_fake_av_celeb_dataloader(
        root_dir=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loss = validate_model(model, val_loader, device)

    print(loss)  # {'val_loss': 1.2307684506688799}


if __name__ == "__main__":
    set_start_method_spawn()

    main()

    # model = load_pretrained_model("ir_50")
    # test_image_path = "./AdaFace/face_alignment/test_images"
    # features = []
    # for fname in sorted(os.listdir(test_image_path)):
    #     path = os.path.join(test_image_path, fname)
    #     aligned_rgb_img = align.get_aligned_face(path)
    #     bgr_tensor_input = to_input(aligned_rgb_img)
    #     feature, _ = model(bgr_tensor_input)
    #     features.append(feature)

    # similarity_scores = torch.cat(features) @ torch.cat(features).T
    # print(similarity_scores)
