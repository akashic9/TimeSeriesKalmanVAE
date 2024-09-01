import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from moviepy.editor import ImageSequenceClip
from torchmetrics.image import StructuralSimilarityIndexMeasure


def ssim(
    image: torch.Tensor,
    reconstructed_image: torch.Tensor,
    observation_mask: torch.Tensor,
):
    T, B, C, H, W = (
        image.shape
    )  # T: sequence length, B: batch size, C: channels, H: height, W: width
    reconstructed_image = torch.tensor(reconstructed_image).to(image.device)
    image_range = image.max() - image.min()
    reconstructed_range = reconstructed_image.max() - reconstructed_image.min()
    _range = torch.max(image_range, reconstructed_range)
    _range_ceil = int(torch.ceil(_range).cpu().detach().numpy())
    _ssim = StructuralSimilarityIndexMeasure(data_range=_range_ceil).to(image.device)
    image_reshape = image.reshape(-1, C, H, W)
    reconstructed_image_reshape = reconstructed_image.reshape(-1, C, H, W)
    ssim_values = torch.zeros(T * B, C).to(image.device)
    for ch in range(C):
        ssim_values[:, ch] = _ssim(
            image_reshape[:, ch, :, :].unsqueeze(1),
            reconstructed_image_reshape[:, ch, :, :].unsqueeze(1),
        )
    ssim_mean = ssim_values.reshape(T, B, C).mean(axis=1)
    observation_mask = observation_mask.expand(-1, 2)
    incorrect = ssim_mean * (1 - observation_mask)
    return incorrect.sum(axis=0) / (1.0 - observation_mask).sum(axis=0)


class EvaluationManager:
    def __init__(
        self,
        dataloader,
        model,
        sample_control,
        transform_handler,
        config,
    ):
        # self.dataloader = dataloader
        self.model = model
        self.sample_control = sample_control
        self.transform_handler = transform_handler
        self.config = config
        self.evaluate_data = next(iter(dataloader)).to(
            device=self.config.device, dtype=self.config.precision
        )
        self.evaluate_sequence_length = self.evaluate_data.shape[0]
        self.window_step = (
            self.transform_handler.window - self.transform_handler.overlap
        )
        self.cut_length = 2

    def window2indices(self, window):
        return (
            self.transform_handler.window
            - self.cut_length * 2
            + self.window_step * (window - 1)
        )

    def get_images(self, data, mask=None):
        _, info = self.model.elbo(
            xs=data,
            observation_mask=mask,
            sample_control=self.sample_control,
        )

        seq_length, batch_size, image_channels, *image_size = data.shape
        filtered_images = (
            self.model.decoder(info["filter_as"].view(-1, self.config.a_dim))
            .mean.view(seq_length, batch_size, image_channels, *image_size)
            .cpu()
            .float()
            .detach()
            .numpy()
        )
        smoothed_images = (
            self.model.decoder(info["as_resampled"].view(-1, self.config.a_dim))
            .mean.view(seq_length, batch_size, image_channels, *image_size)
            .cpu()
            .float()
            .detach()
            .numpy()
        )
        return info, filtered_images, smoothed_images

    def make_video(self, filename, mask=None, fps=10):
        info, filtered_images, smoothed_images = self.get_images(
            self.evaluate_data[:, 0:1], mask
        )

        idx = 0
        cmap = plt.get_cmap("tab10")
        channels = self.config.scalogram_channels
        channel_length = len(channels)
        a_dim_0 = self.config.latent_channels[0]
        a_dim_1 = self.config.latent_channels[1]
        frame_size = (1200, 400 * (channel_length + 1))
        frame_paths = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.info(f"Making video {filename.split('/')[-1]}")
            for step, (image) in enumerate(self.evaluate_data[:, 0:1]):
                fig, axes = plt.subplots(
                    figsize=(frame_size[0] / 100, frame_size[1] / 100),
                    nrows=channel_length + 1,
                    ncols=3,
                )
                fig.suptitle(f"$t = {step}$")
                image = image.cpu().float().detach().numpy()
                for i_axes, _channel in enumerate(channels):
                    axes[i_axes, 0].imshow(
                        image[idx][_channel], aspect="equal", cmap="jet"
                    )
                    axes[i_axes, 1].imshow(
                        filtered_images[step, idx, _channel], aspect="equal", cmap="jet"
                    )
                    axes[i_axes, 2].imshow(
                        smoothed_images[step, idx, _channel], aspect="equal", cmap="jet"
                    )

                axes[channel_length, 0].plot(
                    info["as"][:, idx, a_dim_0].cpu().detach().numpy(),
                    info["as"][:, idx, a_dim_1].cpu().detach().numpy(),
                    ".-",
                    color=cmap(0),
                    label="Encoded",
                )

                axes[channel_length, 0].plot(
                    info["filter_as"][:, idx, a_dim_0].cpu().detach().numpy(),
                    info["filter_as"][:, idx, a_dim_1].cpu().detach().numpy(),
                    ".-",
                    color=cmap(1),
                    label="Filtered",
                )

                axes[channel_length, 0].plot(
                    info["as_resampled"][:, idx, a_dim_0].cpu().detach().numpy(),
                    info["as_resampled"][:, idx, a_dim_1].cpu().detach().numpy(),
                    ".-",
                    color=cmap(2),
                    label="Smoothed",
                )

                for key in ("as", "filter_as", "as_resampled"):
                    axes[channel_length, 0].plot(
                        info[key][step, idx, a_dim_0].cpu().detach().numpy(),
                        info[key][step, idx, a_dim_1].cpu().detach().numpy(),
                        "o",
                        markersize=8,
                        color="red",
                        linestyle="none",
                        zorder=10,
                    )

                axes[channel_length, 1].bar(
                    [str(i) for i in range(self.model.state_space_model.K)],
                    info["weights"][step, idx].cpu().detach().numpy(),
                )
                axes[channel_length, 1].set_ylim(0, 1)
                original_curve = self.transform_handler.wavelet.inverse_transform(
                    image[idx][0], image[idx][1]
                )
                filtered_curve = self.transform_handler.wavelet.inverse_transform(
                    filtered_images[step, idx, 0], filtered_images[step, idx, 1]
                )
                smoothed_curve = self.transform_handler.wavelet.inverse_transform(
                    smoothed_images[step, idx, 0], smoothed_images[step, idx, 1]
                )
                axes[channel_length, 2].plot(original_curve, label="Encoded")
                axes[channel_length, 2].plot(filtered_curve, label="Filtered")
                axes[channel_length, 2].plot(smoothed_curve, label="Smoothed")

                for axes_i, _channel in enumerate(channels):
                    axes[axes_i, 0].set_title(
                        "Original $\\mathbf{x}$ channel %d" % _channel
                    )
                    axes[axes_i, 1].set_title(
                        "Filtered $\\mathbf{x}$ channel %d" % _channel
                    )
                    axes[axes_i, 2].set_title(
                        "Smoothed $\\mathbf{x}$ channel %d" % _channel
                    )
                axes[channel_length, 0].set_title(
                    "$\\mathbf{a}$ space channels %s" % self.config.latent_channels
                )
                axes[channel_length, 0].legend(loc="upper right")
                axes[channel_length, 0].grid()
                axes[channel_length, 1].set_title("Mixture weights")
                axes[channel_length, 2].legend(loc="upper right")
                axes[channel_length, 2].set_title("Time series reconstruction")
                axes[channel_length, 2].set_xlabel("Time")
                axes[channel_length, 2].set_ylabel("Value")

                plt.tight_layout()
                # frame_path = os.path.join(tmpdirname, f"frame_{step:04d}.png")
                frame_path = f"{tmpdirname}/frame_{step:04d}.png"

                fig.savefig(frame_path, dpi=100)
                frame_paths.append(frame_path)
                plt.close(fig)

            video_clip = ImageSequenceClip(frame_paths, fps=fps)
            video_clip.write_videofile(filename, codec="libx264", logger=None)
            logger.info(f"Video saved at {filename}")

    def make_curve(self, filename, mask=None):
        info, filtered_images, smoothed_images = self.get_images(
            self.evaluate_data[:, 0:1], mask
        )

        idx = 0
        logger.info(f"Making curve {filename.split('/')[-1]}")
        original_curves = []
        filtered_curves = []
        smoothed_curves = []
        for step, (image) in enumerate(self.evaluate_data[:, 0:1]):
            image = image.cpu().float().detach().numpy()
            original_curve = self.transform_handler.wavelet.inverse_transform(
                image[idx][0], image[idx][1]
            )
            filtered_curve = self.transform_handler.wavelet.inverse_transform(
                filtered_images[step, idx, 0], filtered_images[step, idx, 1]
            )
            smoothed_curve = self.transform_handler.wavelet.inverse_transform(
                smoothed_images[step, idx, 0], smoothed_images[step, idx, 1]
            )
            original_curves.append(original_curve)
            filtered_curves.append(filtered_curve)
            smoothed_curves.append(smoothed_curve)

        mask_np = mask.cpu().detach().numpy()
        mask_indices = np.where(mask_np == 0)[0]
        eval_window_start = mask_indices[0]
        eval_window_end = mask_indices[-1]
        window_length = len(filtered_curves[0][self.cut_length : -self.cut_length])
        eval_indices_start = self.window2indices(eval_window_start)
        eval_indices_end = self.window2indices(eval_window_end)
        eval_interval_start = (
            self.window2indices(eval_window_start) + window_length - self.window_step
        )
        eval_interval_end = self.window2indices(eval_window_end) + self.window_step
        # eval_window = int(mask.sum().cpu().detach().numpy())
        state_length = len(original_curves)
        curve = original_curves[0][self.cut_length : -self.cut_length]
        for i in range(1, state_length):
            curve_next = original_curves[i][self.cut_length : -self.cut_length]
            curve = np.concatenate(
                (
                    curve,
                    curve_next[-self.window_step :]
                    + (curve[-1] - curve_next[-self.window_step - 1]),
                )
            )

        filtered = []
        for i in range(eval_window_start, eval_window_end + 1):
            current_window = filtered_curves[i][self.cut_length : -self.cut_length]
            current_indices_start = self.window2indices(i)
            tail_diff = eval_interval_end - current_indices_start
            head_diff = eval_interval_start - current_indices_start
            obv_start = max(0, head_diff)
            obv_end = min(window_length, tail_diff)
            obv_value = current_window[obv_start:obv_end]
            if obv_start == 0:
                obv_pre_nan_padding = np.full(-head_diff, np.nan)
                obv_value = np.concatenate((obv_pre_nan_padding, obv_value))
            if obv_end == window_length:
                obv_post_nan_padding = np.full(tail_diff - window_length, np.nan)
                obv_value = np.concatenate((obv_value, obv_post_nan_padding))
            filtered.append(obv_value)
        filtered = np.array(filtered)
        filtered_mean = np.nanmean(filtered, axis=0)
        estimated_error = curve[eval_indices_start] - filtered_mean[0]
        filtered_mean = filtered_mean + estimated_error
        filtered_max = np.nanmax(filtered, axis=0)
        filtered_max = filtered_max + estimated_error
        filtered_min = np.nanmin(filtered, axis=0)
        filtered_min = filtered_min + estimated_error

        smoothed = []
        for i in range(eval_window_start, eval_window_end + 1):
            current_window = smoothed_curves[i][self.cut_length : -self.cut_length]
            current_indices_start = self.window2indices(i)
            tail_diff = eval_interval_end - current_indices_start
            head_diff = eval_interval_start - current_indices_start
            obv_start = max(0, head_diff)
            obv_end = min(window_length, tail_diff)
            obv_value = current_window[obv_start:obv_end]
            if obv_start == 0:
                obv_pre_nan_padding = np.full(-head_diff, np.nan)
                obv_value = np.concatenate((obv_pre_nan_padding, obv_value))
            if obv_end == window_length:
                obv_post_nan_padding = np.full(tail_diff - window_length, np.nan)
                obv_value = np.concatenate((obv_value, obv_post_nan_padding))
            smoothed.append(obv_value)
        smoothed = np.array(smoothed)
        smoothed_mean = np.nanmean(smoothed, axis=0)
        estimated_error = curve[eval_indices_start] - smoothed_mean[0]
        smoothed_mean = smoothed_mean + estimated_error
        smoothed_max = np.nanmax(smoothed, axis=0)
        smoothed_max = smoothed_max + estimated_error
        smoothed_min = np.nanmin(smoothed, axis=0)
        smoothed_min = smoothed_min + estimated_error

        x1 = np.arange(len(curve))
        x2 = np.arange(
            eval_indices_start,
            eval_indices_start + self.config.evaluation_continuous_mask_curve,
        )
        fig, axes = plt.subplots(
            figsize=(10, 10),
            nrows=2,
            ncols=1,
        )
        fig.suptitle("Curve Reconstruction")
        axes[0].plot(x1, curve, label="Encoded")
        axes[0].plot(x2, filtered_mean, label="Filtered")
        axes[0].fill_between(
            x2,
            filtered_min,
            filtered_max,
            color="gray",
            alpha=0.3,
            label="Filtered Range",
        )
        axes[0].legend()
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Value")

        axes[1].plot(x1, curve, label="Encoded")
        axes[1].plot(x2, smoothed_mean, label="Smoothed")
        axes[1].fill_between(
            x2,
            smoothed_min,
            smoothed_max,
            color="gray",
            alpha=0.3,
            label="Smoothed Range",
        )
        axes[1].legend()
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Value")
        plt.savefig(filename)

    def calculate_incorrect_pixels(self, mask=None):
        info, filtered_images, smoothed_images = self.get_images(
            self.evaluate_data, mask
        )
        filtering_incorrect_pixel = (
            ssim(self.evaluate_data, filtered_images, mask)
            .cpu()
            .detach()
            .numpy()
            .T.tolist()
        )
        smoothing_incorrect_pixel = (
            ssim(self.evaluate_data, smoothed_images, mask)
            .cpu()
            .detach()
            .numpy()
            .T.tolist()
        )
        return filtering_incorrect_pixel, smoothing_incorrect_pixel

    def create_continuous_mask(self, mask_length, end=False):
        lst = [1.0] * self.evaluate_sequence_length
        if end:
            start_index = self.evaluate_sequence_length - mask_length
        else:
            start_index = (self.evaluate_sequence_length - mask_length) // 2
        for i in range(start_index, start_index + mask_length):
            lst[i] = 0.0
        return (
            torch.tensor(lst)
            .repeat(1, 1)
            .transpose(0, 1)
            .to(device=self.config.device, dtype=self.config.precision)
        )

    def create_random_mask(self, mask_rate):
        mask = (
            torch.rand((self.evaluate_sequence_length, 1), device=self.config.device)
            >= mask_rate
        ).to(device=self.config.device, dtype=self.config.precision)
        mask[0] = 1
        mask[-1] = 1
        return mask

    def log_unmasked_video(self, video_path):
        # batch = next(iter(self.dataloader))
        # batch = batch.to(dtype=self.config.precision).to(self.config.device)
        # seq_length, batch_size, image_channels, *image_size = batch.shape
        file_name = "unmasked.mp4"
        video_name = f"{video_path}/{file_name}"
        self.make_video(
            filename=video_name,
            fps=10,
        )
        video = wandb.Video(
            video_name,
            file_name,
            fps=10,
            format="mp4",
        )
        wandb.log({"unmasked_video": video})

    def log_continuous_masked_video(self, video_path):
        for mask_length in self.config.evaluation_continuous_mask_video:
            mask = self.create_continuous_mask(
                mask_length=mask_length,
            )
            file_name = f"continuous_mask_{mask_length}.mp4"
            video_name = f"{video_path}/{file_name}"
            self.make_video(
                mask=mask,
                filename=video_name,
                fps=10,
            )
            video = wandb.Video(
                video_name,
                file_name,
                fps=10,
                format="mp4",
            )
            wandb.log({"continuous_masked_video": video})

    def log_random_masked_video(self, video_path):
        for mask_rate in self.config.evaluation_random_mask_video:
            mask = self.create_random_mask(
                mask_rate=mask_rate,
            )
            file_name = f"random_mask_{mask_rate}.mp4"
            video_name = f"{video_path}/{file_name}"
            self.make_video(
                mask=mask,
                filename=video_name,
                fps=10,
            )
            video = wandb.Video(
                video_name,
                file_name,
                fps=10,
                format="mp4",
            )
            wandb.log({"random_masked_video": video})

    def log_continuous_masked_curve(self, image_path):
        file_name = "curve_reconstruction.png"
        image_name = f"{image_path}/{file_name}"
        logger.info(f"Making {file_name}")

        mask_length = (
            int(
                (
                    (self.transform_handler.window - 2 * 2)
                    + self.config.evaluation_continuous_mask_curve
                )
                / (self.transform_handler.window - self.transform_handler.overlap)
            )
            - 1
        )
        mask = self.create_continuous_mask(mask_length=mask_length)
        self.make_curve(
            mask=mask,
            filename=image_name,
        )
        wandb.log({"continuous_masking_estimate": wandb.Image(image_name)})

    def log_continuous_masking_table(self, csv_path):
        file_name = "continuous_masking.csv"
        csv_name = f"{csv_path}/{file_name}"
        logger.info(f"Making {file_name}")
        mask_lengths = np.arange(2, self.evaluate_sequence_length - 4, 2).tolist()
        filtering_incorrect_pixels = []
        smoothing_incorrect_pixels = []
        for mask_length in mask_lengths:
            # seq_length, batch_size, image_channels, *image_size = batch.shape
            mask = self.create_continuous_mask(
                mask_length=mask_length,
            )
            filtering_incorrect_pixel, smoothing_incorrect_pixel = (
                self.calculate_incorrect_pixels(mask)
            )
            filtering_incorrect_pixels.append(filtering_incorrect_pixel)
            smoothing_incorrect_pixels.append(smoothing_incorrect_pixel)
        filter_len = len(filtering_incorrect_pixels[0])
        smooth_len = len(smoothing_incorrect_pixels[0])
        filtering_columns = {
            f"filtering_incorrect_pixels_channel{ch}": [
                row[ch] for row in filtering_incorrect_pixels
            ]
            for ch in range(filter_len)
        }
        smoothing_columns = {
            f"smoothing_incorrect_pixels_channel{ch}": [
                row[ch] for row in smoothing_incorrect_pixels
            ]
            for ch in range(smooth_len)
        }
        table = pd.DataFrame(
            {
                "mask_length": mask_lengths,
                **filtering_columns,
                **smoothing_columns,
            }
        )
        table.to_csv(csv_name)
        logger.info(f"Table saved at {csv_name}")
        wandb.log({"continuous_masking": wandb.Table(dataframe=table)})

    def log_random_masked_table(self, csv_path):
        file_name = "random_masking.csv"
        csv_name = f"{csv_path}/{file_name}"
        logger.info(f"Making {file_name}")
        dropout_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        filtering_incorrect_pixels = []
        smoothing_incorrect_pixels = []
        for dropout_probability in dropout_probabilities:
            # seq_length, batch_size, image_channels, *image_size = batch.shape
            mask = self.create_random_mask(
                mask_rate=dropout_probability,
            )
            filtering_incorrect_pixel, smoothing_incorrect_pixel = (
                self.calculate_incorrect_pixels(mask)
            )
            filtering_incorrect_pixels.append(filtering_incorrect_pixel)
            smoothing_incorrect_pixels.append(smoothing_incorrect_pixel)
        filter_len = len(filtering_incorrect_pixels[0])
        smooth_len = len(smoothing_incorrect_pixels[0])
        filtering_columns = {
            f"filtering_incorrect_pixels_channel{ch}": [
                row[ch] for row in filtering_incorrect_pixels
            ]
            for ch in range(filter_len)
        }
        smoothing_columns = {
            f"smoothing_incorrect_pixels_channel{ch}": [
                row[ch] for row in smoothing_incorrect_pixels
            ]
            for ch in range(smooth_len)
        }
        table = pd.DataFrame(
            {
                "dropout_probability": dropout_probabilities,
                **filtering_columns,
                **smoothing_columns,
            }
        )
        table.to_csv(csv_name)
        logger.info(f"Table saved at {csv_name}")
        wandb.log({"random_masking": wandb.Table(dataframe=table)})

    def __call__(self, epoch):
        if self.config.evaluation_interval_video > 0:
            if epoch % self.config.evaluation_interval_video == 0:
                logger.info("Evaluating video...")
                video_path_root = f"{self.config.checkpoint_dir}/videos/epoch_{epoch}"
                os.makedirs(video_path_root, exist_ok=True)
                self.model.eval()
                if self.config.unmasked_video:
                    self.log_unmasked_video(
                        video_path=video_path_root,
                    )
                if self.config.continuous_mask_video:
                    self.log_continuous_masked_video(
                        video_path=video_path_root,
                    )
                if self.config.random_mask_video:
                    self.log_random_masked_video(
                        video_path=video_path_root,
                    )

        if self.config.evaluation_interval_table > 0:
            if epoch % self.config.evaluation_interval_table == 0:
                logger.info("Evaluating table...")
                csv_path_root = f"{self.config.checkpoint_dir}/csv/epoch_{epoch}"
                os.makedirs(csv_path_root, exist_ok=True)
                self.model.eval()
                if self.config.continuous_mask_table:
                    self.log_continuous_masking_table(
                        csv_path=csv_path_root,
                    )
                if self.config.random_mask_table:
                    self.log_random_masked_table(
                        csv_path=csv_path_root,
                    )
        if self.config.evaluation_interval_curve > 0:
            if epoch % self.config.evaluation_interval_curve == 0:
                logger.info("Evaluating curve...")
                curve_path_root = f"{self.config.checkpoint_dir}/image/epoch_{epoch}"
                os.makedirs(curve_path_root, exist_ok=True)
                self.model.eval()
                if self.config.continuous_mask_curve:
                    self.log_continuous_masked_curve(
                        image_path=curve_path_root,
                    )
