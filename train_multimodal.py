import argparse
import os
import shutil

import torch
from dataset.deepfake_data_multimodal import DeepFakeDataModule
from model.detection_multimodal import MMDFDetection
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributed import destroy_process_group
from utils.utils import read_config

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Simple classification training")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str, default="./dataset")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--margin", type=float, default=None)


def train(args, config):
    gpus = args.gpus
    if args.margin is not None:
        config.loss.margin = args.margin
        config.name += f"_m{args.margin}"

    video_size = config.model.video_encoder.vid_size
    dm = DeepFakeDataModule(
        dataset_name=config.dataset_name,
        frame_padding=config.num_frames,
        batch_size=args.batch_size,
        video_size=(video_size, video_size),
        num_workers=args.num_workers,
        take_train=args.num_train,
        take_val=args.num_val,
        take_test=0,
        to_be_ignored=config.to_be_ignored,
        modality=config.modality,
        # ignored_wav2lip_cluster=config.ignored_wav2lip_cluster,
    )

    model = MMDFDetection(
        config,
        distributed=gpus > 1,
    )

    monitor = "val_loss"
    dirpath = f"./ckpt_{config.dataset_name}/{config.name}"
    os.makedirs(dirpath, exist_ok=True)
    log_path = f"./logs_{config.dataset_name}"

    logger = TensorBoardLogger(
        save_dir=".",
        version=config.name,
        name=log_path,
        default_hp_metric=False,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=config.optimizer.early_stopping,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        save_last=False,
        filename=config.name + "-{epoch}-{val_loss:.3f}",
        monitor=monitor,
        mode="min",
        save_top_k=2,  # -1,
    )
    trainer = Trainer(
        log_every_n_steps=20,
        max_epochs=config.optimizer.max_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            early_stop_callback,
        ],
        enable_checkpointing=True,
        benchmark=True,
        accelerator="auto",
        devices=gpus,
        strategy=None if gpus < 2 else "ddp",
        resume_from_checkpoint=args.resume,
        logger=logger,
    )
    shutil.copyfile(args.config, f"{dirpath}/config.toml")
    trainer.fit(model, dm)
    destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)
    train(args, config)
