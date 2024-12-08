import torch
from data_loader.vox_celeb2.VoxCeleb2Ada import (
    create_train_dict,
    idx_to_speaker_id,
    speaker_id_to_idx,
)
from torch.nn import functional as F
from torchvision.io import read_video

path = "datasets/train"
# video_path = "datasets/train/id00019/_lmvY4AiroM/00121.mp4"

# video, _, _ = read_video(video_path)

# print(video.shape)


batch_videos = torch.randn(30, 64, 64, 3)

padding = (
    0,
    0,
    0,
    0,
    0,
    0,
    10,
    0,
)  # Format: (left, right, top, bottom, front, back, d1_front, d1_back)

padded_videos = F.pad(batch_videos, padding, value=0)

print(padded_videos.shape)

# train_dict = create_train_dict(path)

# keys = list(train_dict.keys())
# print("--------------------------------\n")

# print(train_dict[keys[0]])
# print("--------------------------------\n")

# scene_keys = list(train_dict[keys[0]].keys())
# print("--------------------------------\n")

# print(scene_keys[0])

# print("--------------------------------\n")

# print(train_dict[keys[0]][scene_keys[0]])
# print("--------------------------------\n")


# print(speaker_id_to_idx(train_dict))

# print("--------------------------------\n")


# print(idx_to_speaker_id(train_dict))
