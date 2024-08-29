import os
from dataclasses import dataclass
from typing import Literal
from typing import Tuple

import numpy as np
import torch
from loguru import logger

logger.level("PROGRESS", no=25, color="<green>")


def progress_bar(iterable, func_name="udf", bar_length=40):
    total_iterations = len(iterable)

    log = logger.bind(function=func_name)

    for i, item in enumerate(iterable):
        progress_percentage = (i + 1) / total_iterations
        filled_length = int(bar_length * progress_percentage)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        progress_info = (
            f"{i + 1}/{total_iterations} [{bar}] {progress_percentage * 100:.2f}%"
        )
        log.opt(depth=1).log("PROGRESS", f"{func_name} - {progress_info}")
        yield item


def compute_conv2d_output_size(input_size, kernel_size, stride, padding):
    h, w = input_size
    h_out = (h - kernel_size + 2 * padding) // stride + 1
    w_out = (w - kernel_size + 2 * padding) // stride + 1

    return h_out, w_out


def validate_shape(value: torch.Tensor, expected_shape: Tuple[int, ...], name: str):
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, but got {value.shape}"
        )


def aggregate(
    value: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    sequence_operation: Literal["mean", "sum"],
    batch_operation: Literal["mean", "sum"],
):
    validate_shape(value, (sequence_length, batch_size), "value")
    if sequence_operation == "mean":
        value = value.mean(dim=0)
    elif sequence_operation == "sum":
        value = value.sum(dim=0)
    else:
        raise ValueError(f"Invalid sequence operation {sequence_operation}")
    if batch_operation == "mean":
        value = value.mean(dim=0)
    elif batch_operation == "sum":
        value = value.sum(dim=0)
    else:
        raise ValueError(f"Invalid batch operation {batch_operation}")
    return value


def sequence_first_collate_fn(batch: list) -> torch.Tensor:
    """
    Collate function for the DataLoader.
    """
    data = torch.Tensor(np.stack(batch, axis=0))
    data = data.permute(1, 0, 2, 3, 4)
    return data


def get_image_shape(dataloader):
    first_batch = None
    for batch in dataloader:
        first_batch = batch
        break
    image_shape = first_batch.shape[3:]
    image_channels = first_batch.shape[2]
    return image_shape, image_channels


class CheckpointManager:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        config,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.best_loss = float("inf")

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def save(self, epoch, train_loss, test_loss):
        if epoch % self.config.checkpoint_interval == 0 or not self.config.save_best:
            checkpoint_path = f"{self.config.checkpoint_dir}/state-{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                checkpoint_path,
            )
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

        if test_loss < self.best_loss:
            self.best_loss = test_loss
            best_checkpoint_path = f"{self.config.checkpoint_dir}/best_checkpoint.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                best_checkpoint_path,
            )
            logger.debug(f"Update best checkpoint to {best_checkpoint_path}")

    def load_best_checkpoint(self):
        best_checkpoint_path = f"{self.config.checkpoint_dir}/best_checkpoint.pth"
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info(
                f"Loaded best checkpoint from {best_checkpoint_path} (epoch {checkpoint['epoch']})"
            )
            return checkpoint
        else:
            logger.info("No best checkpoint found.")
            return None


@dataclass
class SampleControl:
    encoder: Literal["sample", "mean"] = "sample"
    decoder: Literal["sample", "mean"] = "mean"
    state_transition: Literal["sample", "mean"] = "sample"
    observation: Literal["sample", "mean"] = "sample"
