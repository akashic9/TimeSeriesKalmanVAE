## Configuration Documentation

### 1. **Data Settings**
   - **`data_root_dir`:** The root directory path where the training and testing datasets are located.
   - **`batch_size`:** The number of samples per batch during training.

### 2. **Model Settings**
   - **`z_dim`:** The dimension of the latent state in LGSSM.
   - **`a_dim`:** The dimension of the latent encoding in VAE.
   - **`K`:** Number of modes in dynamics parameter network. 
   - **`codec_type`:** The type of encoder and decoder used in the model.
   - **`reconst_weight`:** The weight of the reconstruction loss.
   - **`regularization_weight`:** The weight of the regularization loss.
   - **`kalman_weight`:** The weight of the Kalman observation, state transition and posterior loss.
   - **`kl_weight`:** The weight of the KL divergence loss.
   - **`learn_noise_covariance`:**  noise matrices **Q** and **R** are learned or fixed.
   - **`initial_noise_scale`:**  The initial scale of the **Q** and **R**.
   - **`init_transition_reg_weight`:**  The regularization weight for the state transition matrix **A**.
   - **`init_observation_reg_weight`:**  The regularization weight for the emission matrix **C**.
   - **`symmetrize_covariance`:**  Controls the symmetry of the covariance matrices in LGSSM.

### 3. **Training Settings**
   - **`epochs`:** The total number of training epochs.
   - **`warmup_epochs`:** A period at the beginning of training that the weights used in the dynamic parameters of the model are not updated by the observations.
   - **`learning_rate`:** The initial learning rate for the optimizer.
   - **`learning_rate_decay`:** The decay rate of the learning rate.
   - **`burn_in`:** A period at the beginning of a time series during which the model's estimates are not updated by the observations.
   - **`batch_operation`:** The aggregation operation over batches.
   - **`sequence_operation`:** The aggregation operation over sequences.
   - **`scheduler_step`:** The step interval for the learning rate scheduler.
   - **`run_mode`:** Determines whether the model tests on the entire test dataset after training for an epoch (`epoch-wise`) or alternates between training and testing batch by batch (`batch-wise`).
### 4. **Evaluation Settings**
   - **`evaluation`:** Whether to evaluate the model during training.
   - **`evaluation_interval_video`:** The interval at which the model is visualized during training.
   - **`evaluation_interval_table`:** The interval at which the model is evaluated during training.
   - **`continuous_mask_video`:** Whether to make videos using a continuous mask to evaluate the model.
   - **`random_mask`:** Whether to make videos using a random mask to evaluate the model.
   - **`continuous_mask_table`:** Whether to make tables using a continuous mask to evaluate the model.
   - **`random_mask_table`:** Whether to make tables using a random mask to evaluate the model.
### 5. **Plot Settings**
   - **`scalogram_channels`:**  Selects which channels to plot in the scalogram.
   - **`latent_channels`:** Selects which dimensions of the latent encoding **a** to plot.

### 6. **Environment Settings**
   - **`device`:** The device used for training the model.
   - **`precision`** `double` for float64, `single` for float32
   - **`amp`:** Whether to use automatic mixed precision. If use, force `precision` to `single` and force `device` to `cuda`.
   - **`checkpoint_dir`:** The directory where model checkpoints are saved.
   - **`checkpoint_interval`:** The interval at which the model is saved during training.
   - **`save_best`:** Whether to save the best model based on the test loss.

   - **`use_wandb`:** Whether to log the training process to wandb.
   - **`project_name`:** The name of the project in wandb.
   - **`run_name`:** Automatically generated name for the run in wandb.