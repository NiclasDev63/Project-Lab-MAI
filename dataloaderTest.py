import torch
from AdaFace.inference import load_pretrained_model
from crossmodal_training import create_voxceleb2_dataloader

if __name__ == "__main__":

    dataloader = create_voxceleb2_dataloader(
        root_dir="datasets", split="test", batch_size=2, num_workers=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adaface = load_pretrained_model("ir_50").to(device)

    for batch in dataloader:
        frames = batch["frames"].to(device)
        batch_size, num_frames = frames.shape[:2]
        frame_features = []

        # Process each frame individually through AdaFace
        for i in range(num_frames):
            frame = frames[:, i]  # (batch_size, 3, 112, 112)
            frame = frame.contiguous()
            features = adaface(frame)[0]  # Get identity features
            frame_features.append(features)

        # Stack frame features
        frame_features = torch.stack(
            frame_features, dim=1
        )  # (batch_size, num_frames, 512)
        print(frame_features.shape)
