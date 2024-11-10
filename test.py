import json

import torch
from torchvision.transforms import v2 as T

from data_loader.vox_celeb2.main import (
    VoxCeleb2,
    get_train_list,
    idx_to_speaker_id,
    speaker_id_to_idx,
)
from data_loader.vox_celeb2.video_transforms import (
    NormalizeVideo,
    ResizeVideo,
    SquareVideo,
    ToTensorVideo,
)

train_path = "datasets/test/id00528"
train_list_path = "datasets/train_list.txt"
speaker_idx = 348  # equals speaker id00528
video_transform = T.Compose(
    [
        SquareVideo(),
        ResizeVideo((224, 224), T.InterpolationMode.BICUBIC),
        ToTensorVideo(),
    ]
)
vox_celeb2_dataset = VoxCeleb2(
    data_root="./datasets/test", video_transform=video_transform
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample = vox_celeb2_dataset[speaker_idx]
audio_sample = sample["audio"].to(device)


# train_list = get_train_list("./datasets/test")
# speaker_id = "id00528"
# sample = json.dumps(train_list[speaker_id], indent=4)
# id_to_idx = speaker_id_to_idx(train_list)
# idx_to_id = idx_to_speaker_id(train_list)

# print(id_to_idx[speaker_id])
# print(idx_to_id[id_to_idx[speaker_id]])
