import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import PIL
import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

# from pytorch_lightning import LightningDataModule
# from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as nnF
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchaudio.transforms import Resample
from torchvision.io import write_video
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

from AdaFace.face_alignment import align

# from transformers import AutoImageProcessor
from data_loader.vox_celeb2.video_transforms import (
    NormalizeVideo,
    ResizeVideo,
    SquareVideo,
    ToTensorVideo,
)

MAX_VIDEO_LENGTH_IN_SECONDS = 20
FRAME_RATE = 25
MAX_FRAMES = MAX_VIDEO_LENGTH_IN_SECONDS * FRAME_RATE

# Target length is only 5 Frames, as mentioned by marcel
TARGET_LENGTH = 5


def create_train_dict(data_root):
    return_dict = defaultdict(list)
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".mp4"):
                # Extract components of the path
                relative_path = os.path.relpath(os.path.join(root, file), data_root)

                # Continent / sex / speaker_id / video_id.mp4
                components = relative_path.split(os.sep)

                if len(components) == 4:
                    continent, sex, speaker_id, file_name = components

                    # Construct file_id and video_path
                    file_id = file_name.split(".")[
                        0
                    ]  # Remove file extension for file_id
                    video_path = os.path.join(root, file)

                    # Append to dictionary
                    return_dict[speaker_id].append(
                        {
                            "file_id": file_id,
                            "speaker_id": speaker_id,
                            "video_path": video_path,
                            "continent": continent,
                            "sex": sex,
                        }
                    )
    return return_dict


def speaker_id_to_idx(train_list: dict[str, dict[str, list[dict]]]) -> dict[str, int]:
    return {speaker_id: idx for idx, speaker_id in enumerate(train_list.keys())}


def idx_to_speaker_id(train_list: dict[str, dict[str, list[dict]]]) -> dict[int, str]:
    return {idx: speaker_id for idx, speaker_id in enumerate(train_list.keys())}


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


class FakeAVCeleb(Dataset):

    def __init__(
        self,
        data_root: str,
        frame_size: Tuple[int, int] = (112, 112),
    ):
        self.data_dir = data_root
        self.train_dict = create_train_dict(self.data_dir)
        self.speaker_id_to_idx = speaker_id_to_idx(self.train_dict)
        self.idx_to_speaker_id = idx_to_speaker_id(self.train_dict)
        self.frame_size = frame_size

        # Its important to apply the transforms in this order
        self.video_transforms = transforms.Compose(
            [
                SquareVideo(),
                # ResizeVideo(frame_size, InterpolationMode.BILINEAR), # commented this out, because resizing was moved to "_get_aligned_face"
                ToTensorVideo(max_pixel_value=255.0),
                # use mean and std as described in AdaFace github repo see: https://github.com/mk-minchul/AdaFace?tab=readme-ov-file#general-inference-guideline
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # convert rgb to bgr as described in AdaFace github repo see: https://github.com/mk-minchul/AdaFace?tab=readme-ov-file#general-inference-guideline
                transforms.Lambda(self._convert_rgb_to_bgr),
            ]
        )

    @staticmethod
    def _convert_rgb_to_bgr(x: Tensor) -> Tensor:
        return x[:, [2, 1, 0], :, :]

    def _pad_video(self, video: Tensor, target_length: int = TARGET_LENGTH) -> Tensor:
        """
        Pads a video to have target_length number of frames by repeating the last frame

        Args:
            video (torch.Tensor): Tensor has shape (num_frames, height, width, channels).
            target_length (int): Desired number of frames.

        Returns:
            torch.Tensor: A tensor of shape (target_length, height, width, channels).
        """
        num_frames, height, width, channels = video.shape

        if num_frames >= target_length:
            # If video is longer or equal to target length, just slice
            return video[:target_length]
        else:
            # Create a tensor to hold the padded video
            padded_video = torch.zeros(
                (target_length, height, width, channels),
                dtype=video.dtype,
                device=video.device,
            )

            # Copy original video frames
            padded_video[:num_frames] = video

            # Repeat the last frame to fill the remaining slots
            padded_video[num_frames:] = video[-1]

        return padded_video

    def _get_aligned_face(self, frames: Tensor) -> Tensor:
        # frames shape: (T, H, W, C)
        aligned_frames = []

        for frame in frames:
            # Convert single frame to numpy and then PIL
            img_frame = frame.cpu().numpy().astype(np.uint8)
            img_frame = PIL.Image.fromarray(img_frame)

            # Get aligned face
            aligned_rgb_img = align.get_aligned_face(rgb_pil_image=img_frame)

            # If no face detected, use original frame
            if aligned_rgb_img is None:
                # Rearrange the color channel to match the expected input shape of the resize function
                aligned_rgb_img = rearrange(frame, "h w c -> c h w")

                # We need to resize the frame in order to match the returned shape of the align function, which is (112, 112, 3)
                aligned_rgb_img = F.resize(
                    aligned_rgb_img, self.frame_size, InterpolationMode.BILINEAR
                )
                # Rearrange the color channel, to match the output of the align function
                aligned_rgb_img = rearrange(aligned_rgb_img, "c h w -> h w c")
            else:
                aligned_rgb_img = np.array(aligned_rgb_img)
                aligned_rgb_img = torch.from_numpy(aligned_rgb_img).float()

            aligned_frames.append(aligned_rgb_img)

        return (
            torch.stack(aligned_frames, dim=0)
            if len(aligned_frames) > 0
            else torch.zeros(
                (len(frames) if len(frames) > 0 else TARGET_LENGTH, 112, 112, 3)
            )
        )

    def __getitem__(self, index: int) -> dict:
        sample = self._get_random_speaker_sample(index)

        video_path = sample["video_path"]

        outputs = {}

        video, _, _ = read_video(video_path)
        video = self._get_aligned_face(video)
        video = video.detach()
        video = self._pad_video(video)
        video = rearrange(video, "t h w c -> t c h w")
        video = self.video_transforms(video)

        outputs["frames"] = video
        outputs["file_id"] = sample["file_id"]
        outputs["video_path"] = video_path
        outputs["speaker_id"] = sample["speaker_id"]

        return outputs

    def __len__(self) -> int:
        return len(self.train_dict.keys())

    def _get_random_speaker_sample(self, index: int) -> dict:
        speaker_id = self.idx_to_speaker_id[index]
        speaker_data_meta_data = self.train_dict[speaker_id]
        return speaker_data_meta_data[
            random.randint(0, len(speaker_data_meta_data) - 1)
        ]


def create_fake_av_celeb_dataloader(
    root_dir,
    batch_size=8,
    num_workers=4,
    frame_size=(112, 112),
):
    """
    Create a DataLoader for the FakeAVCeleb dataset
    """
    dataset = FakeAVCeleb(
        data_root=root_dir,
        frame_size=frame_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
