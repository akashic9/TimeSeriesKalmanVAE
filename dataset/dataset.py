import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def calculate_window_indices(total_length, window_length, overlap_length):
    step = window_length - overlap_length
    start_offset = (total_length - window_length) % step
    window_indices = []
    for start in range(start_offset, total_length - window_length + 1, step):
        end = start + window_length
        window_indices.append((start, end))
    return window_indices


def nearest_multiple_of_8(n):
    """
    计算离给定整数最近的8的整数倍

    :param n: 输入的整数
    :return: 离给定整数最近的8的整数倍
    """
    # 计算商
    quotient = n // 8

    # 找到两个最近的8的倍数
    lower_multiple = quotient * 8
    upper_multiple = (quotient + 1) * 8

    # 比较哪个更接近
    if abs(n - lower_multiple) < abs(n - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple


class ComplexMorletWaveletTransform:
    def __init__(
        self,
        scales,
        wavelet_params=(1.5, 0.5),
        decompose="magnitude_phase",
        return_main=True,
        return_sub=True,
    ):
        self.scales = scales
        self.wavelet_params = wavelet_params
        self.decompose = decompose
        self.return_main = return_main
        self.return_sub = return_sub

    def fit_transform(self, X):
        coefficients, _ = pywt.cwt(
            X, self.scales, f"cmor{self.wavelet_params[0]}-{self.wavelet_params[1]}"
        )
        if self.decompose == "magnitude_phase":
            _main = np.abs(coefficients)
            _sub = np.angle(coefficients)
        elif self.decompose == "real_imaginary":
            _main = np.real(coefficients)
            _sub = np.imag(coefficients)
        else:
            _main = coefficients
            _sub = None

        if self.return_main:
            if self.return_sub:
                return _main, _sub
            else:
                return _main
        else:
            if self.return_sub:
                return _sub
            else:
                return None

    def reconstruct_complex(self, mean, sub):
        if self.decompose == "magnitude_phase":
            return mean * (np.cos(sub) + 1j * np.sin(sub))
        elif self.decompose == "real_imaginary":
            return mean + 1j * sub

    def inverse_transform(self, magnitudes, phases):
        coefficients = self.reconstruct_complex(magnitudes, phases)
        reconstructed_signal = np.zeros(coefficients.shape[1])
        total_scales = len(self.scales)

        # 小波函数
        for i, scale in enumerate(self.scales):
            reconstructed_signal += np.real(coefficients[i, :]) / np.sqrt(scale)

        reconstructed_signal *= 1 / total_scales

        reconstructed_signal_normalized = (
            reconstructed_signal - np.min(reconstructed_signal)
        ) / (np.max(reconstructed_signal) - np.min(reconstructed_signal))
        return reconstructed_signal_normalized


class Scalogram:
    def __init__(self, window, overlap, wavelet):
        self.window = window
        self.overlap = overlap
        self.wavelet = wavelet

    def data2scalogram(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        _data = (data - np.min(data)) / (np.max(data) - np.min(data))
        _data = np.nan_to_num(_data, nan=0)

        if self.wavelet.return_main:
            if self.wavelet.return_sub:
                magnitude, phase = self.wavelet.fit_transform(_data)
                scalograms = [magnitude, phase]
            else:
                magnitude = self.wavelet.fit_transform(_data)
                scalograms = [magnitude]
        else:
            if self.wavelet.return_sub:
                phase = self.wavelet.fit_transform(_data)
                scalograms = [phase]
            else:
                scalograms = []

        scalograms = np.array(scalograms)
        return scalograms.transpose((1, 2, 0))

    def scalogram2data(self, scalogram):
        reconstructed_data = self.wavelet.inverse_transform(
            scalogram[:, :, 0], scalogram[:, :, 1]
        )
        return reconstructed_data

    def generate_scalograms(self, source):
        window_indices = calculate_window_indices(
            len(source), self.window, self.overlap
        )
        images = []
        # images_time = []
        for window_index in window_indices:
            # window_time_interval = source[window_index[0] : window_index[1]]["datetime"]
            # image_time = [window_time_interval.min(), window_time_interval.max()]
            _slice = source[window_index[0] : window_index[1]]
            _image = self.data2scalogram(_slice)
            current_shape = _image.shape
            target_shape = (
                nearest_multiple_of_8(current_shape[0]),
                nearest_multiple_of_8(current_shape[1]),
                current_shape[2],
            )
            zoom_factor = (
                target_shape[0] / current_shape[0],
                target_shape[1] / current_shape[1],
                1,
            )
            resized_image = zoom(_image, zoom_factor)
            images.append(resized_image)
            # images_time.append(image_time)
        # images_np = np.array(images)
        # images_np = np.expand_dims(images_np, axis=1)
        return np.array(images)


class TimeSeriesImageDataset(Dataset):
    def __init__(self, data, window_size, stride, image_transform, split="train"):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.image_transform = image_transform
        self.split = split
        self.cached_images = [None] * self.__len__()

    def __len__(self):
        return (len(self.data) - 2 * self.window_size) // self.stride

    def __getitem__(self, idx):
        if self.cached_images[idx] is not None:
            return self.cached_images[idx]
        if self.split == "train":
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size
        elif self.split == "test":
            start_idx = idx * self.stride + self.window_size
            end_idx = start_idx + self.window_size
        else:
            raise ValueError("Invalid split")
        _ts = self.data[start_idx:end_idx]
        _ts = (np.array(_ts) - np.min(_ts)) / (np.max(_ts) - np.min(_ts))
        _images = self.image_transform(_ts)
        _images = _images.transpose((0, 3, 1, 2))
        self.cached_images[idx] = _images
        return _images


if __name__ == "__main__":
    time_series = np.load("dataset/synthetic.npy")
    scales = np.logspace(0.1, 2, num=48)
    WaveletTransformer = ComplexMorletWaveletTransform(
        scales=scales,
        wavelet_params=(1.5, 0.5),
        return_main=True,
        return_sub=True,
        decompose="magnitude_phase",
    )
    ImageTransformer = Scalogram(window=48, overlap=44, wavelet=WaveletTransformer)
    TSDatasetTrain = TimeSeriesImageDataset(
        data=time_series,
        window_size=300,
        stride=150,
        image_transform=ImageTransformer.generate_scalograms,
        split="train",
    )
    TSDatasetTest = TimeSeriesImageDataset(
        data=time_series,
        window_size=300,
        stride=150,
        image_transform=ImageTransformer.generate_scalograms,
        split="test",
    )
