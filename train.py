import sys

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb
from config.config_loader import training_config_loader
from dataset.dataset import (
    ComplexMorletWaveletTransform,
    Scalogram,
    TimeSeriesImageDataset,
)
from model.evaluation import EvaluationManager
from model.kvae import KalmanVAE
from model.misc import (
    SampleControl,
    sequence_first_collate_fn,
    get_image_shape,
    CheckpointManager,
    progress_bar,
)


class RunEpoch:
    def __init__(
        self,
        config,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        sample_control,
        checkpoint_handler,
        evaluation_handler,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.n_batches_train = len(train_dataloader)
        self.test_dataloader = test_dataloader
        self.n_batches_test = len(test_dataloader)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_control = sample_control
        self.scaler = GradScaler() if self.config.amp else None
        self.checkpoint_handler = checkpoint_handler
        self.evaluation_handler = evaluation_handler
        self.total_loss = None
        self.metrics = None
        self.average_loss = None
        self.loss_reset()

    def loss_reset(self):
        self.total_loss = {"train": 0.0, "test": 0.0}
        self.metrics = {
            "train": {
                "reconstruction": 0.0,
                "regularization": 0.0,
                "kalman_observation_log_likelihood": 0.0,
                "kalman_state_transition_log_likelihood": 0.0,
                "kalman_posterior_log_likelihood": 0.0,
            },
            "test": {
                "reconstruction": 0.0,
                "regularization": 0.0,
                "kalman_observation_log_likelihood": 0.0,
                "kalman_state_transition_log_likelihood": 0.0,
                "kalman_posterior_log_likelihood": 0.0,
            },
        }
        self.average_loss = {"train": 0.0, "test": 0.0}

    def _process(self, data, mode, current_epoch):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        data = data.to(dtype=self.config.precision).to(self.config.device)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(mode == "train"):
            if self.config.amp:
                logger.debug("Using AMP")
                with autocast():
                    elbo, info = self.model.elbo(
                        xs=data,
                        learn_weight_model=(current_epoch >= self.config.warmup_epochs),
                        sample_control=self.sample_control[mode],
                    )
                    loss = -elbo
                    if mode == "train":
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
            else:
                elbo, info = self.model.elbo(
                    xs=data,
                    learn_weight_model=(current_epoch >= self.config.warmup_epochs),
                    sample_control=self.sample_control[mode],
                )
                loss = -elbo
                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
        self.total_loss[mode] += loss.item()
        return elbo, info

    def train(self, current_epoch):
        if self.config.run_mode == "batch-wise":
            for data_train, data_test in zip(
                self.train_dataloader, self.test_dataloader
            ):
                train_elbo, train_info = self._process(
                    data_train, "train", current_epoch
                )
                test_elbo, test_info = self._process(data_test, "test", current_epoch)
                for key_train, key_test in zip(
                    self.metrics["train"], self.metrics["test"]
                ):
                    self.metrics["train"][key_train] += (
                        train_info[key_train] / self.n_batches_train
                    )
                    self.metrics["test"][key_test] += (
                        test_info[key_test] / self.n_batches_test
                    )
            self.average_loss["train"] = self.total_loss["train"] / self.n_batches_train
            self.average_loss["test"] = self.total_loss["test"] / self.n_batches_test
        else:
            for data_train in self.train_dataloader:
                train_elbo, train_info = self._process(
                    data_train, "train", current_epoch
                )
                for key_train in self.metrics["train"]:
                    self.metrics["train"][key_train] += (
                        train_info[key_train] / self.n_batches_train
                    )
            self.average_loss["train"] = self.total_loss["train"] / self.n_batches_train
            for data_test in self.test_dataloader:
                test_elbo, test_info = self._process(data_test, "test", current_epoch)
                for key_test in self.metrics["test"]:
                    self.metrics["test"][key_test] += (
                        test_info[key_test] / self.n_batches_test
                    )
            self.average_loss["test"] = self.total_loss["test"] / self.n_batches_test

    def __call__(self):
        for epoch in progress_bar(
            range(self.config.epochs), func_name="Epoch", bar_length=40
        ):
            self.train(epoch)
            if self.config.evaluation:
                self.evaluation_handler(epoch)
            wandb.log(
                {
                    "train_loss": RE.average_loss["train"],
                    "test_loss": RE.average_loss["test"],
                    "train_metrics": RE.metrics["train"],
                    "test_metrics": RE.metrics["test"],
                }
            )
            if (epoch > 0) & (epoch % configs.scheduler_step == 0):
                self.scheduler.step()
            self.checkpoint_handler.save(
                epoch=epoch,
                train_loss=RE.average_loss["train"],
                test_loss=RE.average_loss["test"],
            )
            self.loss_reset()


if __name__ == "__main__":
    # Load config
    configs = training_config_loader("config/independent_channel_config.yaml")
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"{configs.checkpoint_dir}/record.log", level="INFO")
    # Init wandb
    if configs.use_wandb:
        wandb.init(
            project=configs.project_name,
            name=configs.run_name,
            config=configs.to_dict(),
        )
    else:
        wandb.init(mode="disabled")

    # Get dataloader
    time_series = np.load(configs.data_root_dir)
    scales = np.logspace(0.1, 2, num=48)
    wavelet_transformer = ComplexMorletWaveletTransform(
        scales=scales,
        wavelet_params=(1.5, 0.5),
        return_main=True,
        return_sub=True,
        decompose="real_imaginary",
    )
    sequence_transformer = Scalogram(window=48, overlap=44, wavelet=wavelet_transformer)
    dataset_train = TimeSeriesImageDataset(
        data=time_series,
        window_size=300,
        stride=150,
        image_transform=sequence_transformer.generate_scalograms,
        split="train",
    )
    dataset_test = TimeSeriesImageDataset(
        data=time_series,
        window_size=300,
        stride=150,
        image_transform=sequence_transformer.generate_scalograms,
        split="test",
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=sequence_first_collate_fn,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=configs.batch_size,
        shuffle=False,
        collate_fn=sequence_first_collate_fn,
    )

    shape, channel = get_image_shape(dataloader_train)
    kvae = (
        KalmanVAE(
            image_size=shape,
            image_channels=channel,
            config=configs,
        )
        .to(dtype=configs.precision)
        .to(configs.device)
    )

    adam_optimizer = optim.Adam(kvae.parameters(), lr=configs.learning_rate)
    scheduler_decay = lr_scheduler.ExponentialLR(
        adam_optimizer, gamma=configs.learning_rate_decay
    )

    sample_control_dict = {
        "train": SampleControl(),
        "test": SampleControl(
            encoder="mean", decoder="mean", state_transition="mean", observation="mean"
        ),
    }

    checkpoint_manager = CheckpointManager(
        model=kvae,
        optimizer=adam_optimizer,
        scheduler=scheduler_decay,
        config=configs,
    )
    evaluation_manager = EvaluationManager(
        dataloader=dataloader_test,
        model=kvae,
        sample_control=sample_control_dict["test"],
        transform_handler=wavelet_transformer,
        config=configs,
    )

    RE = RunEpoch(
        config=configs,
        model=kvae,
        optimizer=adam_optimizer,
        sample_control=sample_control_dict,
        train_dataloader=dataloader_train,
        test_dataloader=dataloader_test,
        scheduler=scheduler_decay,
        checkpoint_handler=checkpoint_manager,
        evaluation_handler=evaluation_manager,
    )
    RE()
