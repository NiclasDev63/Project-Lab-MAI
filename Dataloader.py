import torch
from AdaFace.inference import load_pretrained_model
from crossmodal_training import VoxCeleb2Dataset, MultiModalFeatureExtractor
from myadaface import get_identity_videos

extractor = MultiModalFeatureExtractor()
dataset = VoxCeleb2Dataset("datasets", split="test")
model = load_pretrained_model("ir_50")
# identity_videos[person_id] = video_paths
identity_00019_videos = get_identity_videos(split="test")["id00019"]
video_frames = dataset._load_video_frames(identity_00019_videos[0])[0]
all_frames_features = []

for frame in video_frames:
    feature = model(frame.unsqueeze(0))[0]
    all_frames_features.append(feature)
combined_tensor = torch.stack(all_frames_features)
print(combined_tensor.squeeze().shape)
