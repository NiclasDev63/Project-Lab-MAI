import json
import os
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchvision
from einops import rearrange

# from pytorch_lightning import LightningDataModule
# from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchaudio.transforms import Resample
from torchvision.io import write_video
from torchvision.transforms import v2 as T
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim

# from transformers import AutoImageProcessor
from data_loader.vox_celeb2.video_transforms import (
    NormalizeVideo,
    ResizeVideo,
    SquareVideo,
    ToTensorVideo,
)


def _get_padding_pair(padding_size: int, padding_position: str) -> List[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError(
            "Wrong padding position. It should be zero or tail or average."
        )
    return pad


def padding_video(
    tensor: Tensor,
    target: int,
    padding_method: str = "zero",
    padding_position: str = "tail",
) -> Tensor:
    t, _, _, _ = tensor.shape
    padding_size = target - t

    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c h w -> c h w t")
        tensor = F.pad(tensor, pad=pad + [0, 0], mode="replicate")
        return rearrange(tensor, "c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def get_train_list(root: str) -> dict[str, dict[str, list[dict]]]:
    with open(os.path.join(root, "train_list.txt"), "r") as f:
        lines = f.read().splitlines()
        return_dict = {}
        for line in lines:
            # line has the format: id00527 id00527/zrdW6HFWNHM/00092.wav
            audio_path = line.split(" ")[1]
            speaker_id, scene_id, audio_file = audio_path.split("/")
            if speaker_id not in return_dict:
                return_dict[speaker_id] = {}
            if scene_id not in return_dict[speaker_id]:
                return_dict[speaker_id][scene_id] = []
            return_dict[speaker_id][scene_id].append(
                {
                    "file_id": audio_file.replace(".wav", ""),
                    "video_path": audio_path.replace(".wav", ".mp4"),
                    # TODO: change this to .wav
                    "audio_path": audio_path.replace(".wav", ".m4a"),
                }
            )
        return return_dict


def speaker_id_to_idx(train_list: dict[str, dict[str, list[dict]]]) -> dict[str, int]:
    return {speaker_id: idx for idx, speaker_id in enumerate(train_list.keys())}


def idx_to_speaker_id(train_list: dict[str, dict[str, list[dict]]]) -> dict[int, str]:
    return {idx: speaker_id for idx, speaker_id in enumerate(train_list.keys())}


@dataclass
class Metadata:
    file: str
    split: str
    modify_video: bool
    modify_audio: bool
    identity: int
    kind: str
    engine: str
    gesture_type: str
    video_fps: int
    video_frames: int
    audio_fps: int
    audio_frames: int


def read_json(path: str, object_hook=None):
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def read_video(path: str):
    """read video and audio from path

    Args:
        path (str): video path

    Returns:
        (tensor, tensor, dict): video in shape (T, H, W, C), audio in shape (L, K), info
    """
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.permute(1, 0)
    return video, audio, info


class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.

    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats):
        super().__init__()
        self.mean, self.std = stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def __repr__(self):
        format_string = self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"
        return format_string


norm_dict = {
    "iresnet18": {"mean": [0.4782, 0.4367, 0.4287], "std": [0.2246, 0.2154, 0.2084]},
    "swin": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}


class VoxCeleb2(Dataset):

    def __init__(
        self,
        data_root: str,
        video_transform: Optional[Callable],
        frame_padding: int = 512,
        modality: str = "both",
    ):
        self.data_root = data_root
        self.train_list = get_train_list(self.data_root)
        self.speaker_id_to_idx = speaker_id_to_idx(self.train_list)
        self.idx_to_speaker_id = idx_to_speaker_id(self.train_list)
        self.video_padding = frame_padding
        self.video_transform = video_transform
        self.modality = modality

    def __getitem__(self, index: int) -> dict:
        sample = self._get_random_speaker_sample(index)

        video_path = os.path.join(self.data_root, sample["video_path"])
        audio_path = os.path.join(self.data_root, sample["audio_path"])

        outputs = {}

        # TODO: check if video modality handling is correct
        if self.modality in ["video", "both"]:
            video, _, _ = read_video(video_path)
            video = padding_video(video, target=self.video_padding)
            video = rearrange(video, "t h w c -> t c h w")
            video = self.video_transform(video)
            outputs["video"] = video

        if self.modality in ["audio", "both"]:
            audio = load_audio(audio_path)
            audio = pad_or_trim(audio)
            audio = log_mel_spectrogram(audio)
            outputs["audio"] = audio

        outputs["file_id"] = sample["file_id"]

        return outputs

    def __len__(self) -> int:
        return len(self.train_list)

    def _get_random_speaker_sample(self, index: int) -> dict:
        speaker_id = self.idx_to_speaker_id[index]
        speaker_data_meta_data = self.train_list[speaker_id]
        random_scene_id = random.choice(list(speaker_data_meta_data.keys()))
        scene_samples = speaker_data_meta_data[random_scene_id]
        return scene_samples[random.randint(0, len(scene_samples) - 1)]


# class PretrainDataModule(LightningDataModule):
#     metadata: List[Metadata]

#     def __init__(
#         self,
#         data_root: str = "data",
#         frame_padding: int = 128,
#         video_size: Tuple[int, int] = (224, 224),
#         batch_size: int = 1,
#         num_workers: int = 0,
#         take_train: int = None,
#         take_val: int = None,
#         take_test: int = None,
#         return_file_name: bool = False,
#         modality: str = "both",
#     ):
#         super().__init__()
#         self.root = data_root  # insert root here      # get_data_root(dataset_name)
#         self.frame_padding = frame_padding
#         self.batch_size = batch_size
#         self.video_size = video_size
#         self.num_workers = num_workers
#         self.take_train = take_train
#         self.take_val = take_val
#         self.take_test = take_test
#         self.return_file_name = return_file_name
#         self.modality = modality

#     def setup(self, stage: Optional[str] = None) -> None:
#         self.metadata: List[Metadata] = read_json(
#             os.path.join(self.root, "metadata.json"),
#             lambda x: Metadata(**x),
#         )
#         train_metadata, val_metadata, test_metadata = [], [], []

#         for meta in self.metadata:
#             if meta.split == "train":
#                 train_metadata.append(meta)
#             elif meta.split == "val":
#                 val_metadata.append(meta)
#             elif meta.split == "test":
#                 test_metadata.append(meta)

#         if self.take_train is not None:
#             train_metadata = train_metadata[: self.take_train]
#         if self.take_val is not None:
#             val_metadata = val_metadata[: self.take_val]
#         if self.take_test is not None:
#             test_metadata = test_metadata[: self.take_test]

#         norm = NormalizeVideo(
#             mean=norm_dict["iresnet18"]["mean"], std=norm_dict["iresnet18"]["std"]
#         )
#         train_video_transform = [
#             ResizeVideo(
#                 self.video_size, interpolation_mode=T.InterpolationMode.BICUBIC
#             ),
#             T.RandomHorizontalFlip(p=0.5),
#             # T.RandomResizedCrop(
#             #     self.video_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
#             # ),
#             # T.RandomRotation(
#             #     degrees=20, interpolation=T.InterpolationMode.BILINEAR
#             # ),
#             ToTensorVideo(),
#             norm,
#         ]
#         eval_video_transform = [
#             ResizeVideo(
#                 self.video_size, interpolation_mode=T.InterpolationMode.BICUBIC
#             ),
#             ToTensorVideo(),
#             norm,
#         ]
#         if self.root.split("/")[-1].lower() == "kodf":
#             train_video_transform.insert(0, SquareVideo())
#             eval_video_transform.insert(0, SquareVideo())

#         train_video_transform = T.Compose(train_video_transform)
#         eval_video_transform = T.Compose(eval_video_transform)

#         self.train_dataset = PretrainData(
#             "train",
#             self.root,
#             self.frame_padding,
#             video_transform=train_video_transform,
#             metadata=train_metadata,
#             return_file_name=self.return_file_name,
#             modality=self.modality,
#         )
#         self.val_dataset = PretrainData(
#             "val",
#             self.root,
#             self.frame_padding,
#             video_transform=eval_video_transform,
#             metadata=val_metadata,
#             return_file_name=self.return_file_name,
#             modality=self.modality,
#         )
#         self.test_dataset = PretrainData(
#             "test",
#             self.root,
#             self.frame_padding,
#             video_transform=eval_video_transform,
#             metadata=test_metadata,
#             return_file_name=self.return_file_name,
#             modality=self.modality,
#         )

#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#         )

#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def test_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )
