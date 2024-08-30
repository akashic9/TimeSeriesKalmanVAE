from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Union, Optional

import torch
import yaml


@dataclass
class Config:
    # Environment Settings
    device: str
    precision: Literal["single", "double"]
    amp: bool
    checkpoint_dir: str
    checkpoint_interval: int
    save_best: bool
    project_name: str
    use_wandb: bool

    # Plot Settings
    scalogram_channels: Union[int, List[int]]
    latent_channels: List[int]

    # Data Settings
    data_root_dir: str
    batch_size: int
    batch_operation: Literal["mean", "sum"]
    sequence_operation: Literal["mean", "sum"]

    # Model Settings
    a_dim: int
    z_dim: int
    K: int
    codec_type: Literal["shared", "independent"]
    reconst_weight: float
    regularization_weight: float
    kalman_weight: float
    kl_weight: float
    dpn_hidden_size: int
    dpn_num_layers: int
    init_transition_reg_weight: float
    init_observation_reg_weight: float
    symmetrize_covariance: bool
    learn_noise_covariance: bool
    init_noise_scale: float

    # Training Settings
    epochs: int
    warmup_epochs: int
    learning_rate: float
    learning_rate_decay: float
    scheduler_step: int
    burn_in: int
    run_mode: Literal["batch-wise", "epoch-wise"]

    # Evaluation Settings
    evaluation: bool
    evaluation_interval_video: int
    evaluation_interval_table: int
    evaluation_interval_curve: int
    evaluation_continuous_mask_video: Optional[List[int]]
    evaluation_random_mask_video: Optional[List[float]]
    evaluation_continuous_mask_curve: int
    unmasked_video: bool
    continuous_mask_video: bool
    random_mask_video: bool
    continuous_mask_table: bool
    random_mask_table: bool
    continuous_mask_curve: bool

    def to_dict(self):
        return self.__dict__

    def __post_init__(self):
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = f"{self.checkpoint_dir}/{self.run_name}"
        if isinstance(self.scalogram_channels, int):
            self.scalogram_channels = [self.scalogram_channels]
        self.device = torch.device(self.device)

        if self.precision == "single":
            self.precision = torch.float32
        elif self.precision == "double":
            if self.amp:
                self.precision = torch.float32
            else:
                self.precision = torch.float64
        else:
            raise ValueError("precision must be 'single' or 'double'")

        if self.amp:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(self.device)


def training_config_loader(filename):
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)
