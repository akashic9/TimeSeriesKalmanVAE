# TSKVAE Implementation
This repository contains the implementation of the Time Series Kalman Variational Autoencoder (TSKVAE). The TSKVAE model is designed to handle time series data by converting it into scalograms using the Complex Morlet Wavelet Transform (CMOR) and processing each channel independently to avoid interference.

## Usage
The main script to run the TSKVAE model is `train.py`. The script use configuration files to set the hyperparameters of the model. The configuration files are located in the `config` folder. Please see [CONFIG README](config/config_readme.md) for understanding the configuration files. Model losses and metrics are logged using `wandb`. To see the masking results across epochs, please use `performance_analysis.py`.

## Repository Structure
The repository is structured as follows:
```
config/                 # Configuration files and loader
dataset/               # Datasets and generation scripts
models/                # Model implementation
```