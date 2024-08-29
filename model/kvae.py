import torch
import torch.distributions as D
import torch.nn as nn

from .lgssm import StateSpaceModel
from .misc import validate_shape, aggregate
from .vae import ImageDecoder, ImageEncoder, IndependentEncoder, IndependentDecoder


class KalmanVAE(nn.Module):
    def __init__(
        self,
        image_size,
        image_channels,
        config,
    ):
        super(KalmanVAE, self).__init__()
        if config.codec_type == "shared":
            self.encoder = ImageEncoder(image_size, image_channels, config.a_dim)
            self.decoder = ImageDecoder(image_size, image_channels, config.a_dim)
        elif config.codec_type == "independent":
            self.encoder = IndependentEncoder(image_size, config.a_dim)
            self.decoder = IndependentDecoder(image_size, config.a_dim)
        else:
            raise ValueError(f"Invalid codec type {config.codec_type}")
        self.state_space_model = StateSpaceModel(config=config)
        self.a_dim = config.a_dim
        self.z_dim = config.z_dim
        self.config = config
        self.register_buffer("_zero_val", torch.tensor(0.0))

    def elbo(
        self,
        sample_control,
        xs=None,
        as_=None,
        observation_mask=None,
        learn_weight_model=True,
        burn_in=0,
    ):
        if as_ is None and xs is None:
            raise ValueError("Either as_ or xs must be provided")
        elif as_ is None and xs is not None:
            seq_length = xs.shape[0]
            batch_size = xs.shape[1]

            as_distrib = self.encoder(xs.reshape(-1, *xs.shape[2:]))
            if sample_control.encoder == "sample":
                as_ = as_distrib.rsample().view(seq_length, batch_size, self.a_dim)
            elif sample_control.encoder == "mean":
                if self.training:
                    raise ValueError(
                        "Invalid sample control for encoder: {}".format(
                            sample_control.encoder
                        )
                    )
                as_ = as_distrib.mean.view(seq_length, batch_size, self.a_dim)
            else:
                raise ValueError(
                    "Invalid sample control for encoder: {}".format(
                        sample_control.encoder
                    )
                )
        elif as_ is not None and xs is None:
            seq_length = as_.shape[0]
            batch_size = as_.shape[1]
            validate_shape(as_, (seq_length, batch_size, self.a_dim), "as_")
            reconst_weight = 0.0
            regularization_weight = 0.0
            kl_weight = 0.0
        else:
            raise ValueError("Only one of as_ and xs must be provided")

        # Reconstruction objective

        if xs is not None:
            xs_distrib = self.decoder(as_.view(-1, self.a_dim))
            reconstruction_obj = aggregate(
                xs_distrib.log_prob(xs.reshape(-1, *xs.shape[2:]))
                .view(seq_length, batch_size, *xs.shape[2:])
                .sum([-3, -2, -1]),
                sequence_length=seq_length,
                batch_size=batch_size,
                sequence_operation=self.config.sequence_operation,
                batch_operation=self.config.batch_operation,
            )

            # Regularization objective
            # -ln q_\phi(a|x)
            regularization_obj = aggregate(
                as_distrib.log_prob(as_.view(-1, self.a_dim))
                .view(seq_length, batch_size, self.a_dim)
                .sum(-1),
                sequence_length=seq_length,
                batch_size=batch_size,
                sequence_operation=self.config.sequence_operation,
                batch_operation=self.config.batch_operation,
            )

        # Kalman filter and smoother
        (
            filter_means,
            filter_covariances,
            filter_next_means,
            filter_next_covariances,
            mat_As,
            mat_Cs,
            filter_as,
            weights,
        ) = self.state_space_model.kalman_filter(
            as_,
            sample_control=sample_control,
            observation_mask=observation_mask,
            learn_weight_model=learn_weight_model,
            symmetrize_covariance=self.config.symmetrize_covariance,
            burn_in=burn_in,
        )
        means, covariances, zs, as_resampled = self.state_space_model.kalman_smooth(
            as_,
            filter_means=filter_means,
            filter_covariances=filter_covariances,
            filter_next_means=filter_next_means,
            filter_next_covariances=filter_next_covariances,
            mat_As=mat_As,
            mat_Cs=mat_Cs,
            sample_control=sample_control,
            symmetrize_covariance=self.config.symmetrize_covariance,
            burn_in=burn_in,
        )

        # Sample from p_\gamma (z|a,u)
        # Shape of means: (sequence_length, batch_size, z_dim, 1)
        # Shape of covariances: (sequence_length, batch_size, z_dim, z_dim)
        zs_distrib = D.MultivariateNormal(
            means.view(seq_length, batch_size, self.z_dim),
            covariances.view(seq_length, batch_size, self.z_dim, self.z_dim),
        )

        # KL divergence between q_\phi(a|x) and p(z) for VAE validation purposes
        if self.config.kl_weight != 0.0:
            prior_distrib = D.Normal(
                torch.zeros(self.a_dim, dtype=xs.dtype, device=xs.device),
                torch.ones(self.a_dim, dtype=xs.dtype, device=xs.device),
            )
            kl_reg = -aggregate(
                torch.distributions.kl.kl_divergence(as_distrib, prior_distrib)
                .view(seq_length, batch_size, self.a_dim)
                .sum(-1),
                sequence_length=seq_length,
                batch_size=batch_size,
                sequence_operation=self.config.sequence_operation,
                batch_operation=self.config.batch_operation,
            )
        else:
            kl_reg = self._zero_val

        # For testing purposes
        # zs_distrib = D.MultivariateNormal(torch.stack(filter_means).view(-1, self.z_dim), torch.stack(filter_covariances).view(-1, self.z_dim, self.z_dim))

        zs_sample = zs_distrib.rsample()

        # ln p_\gamma(a|z)
        kalman_observation_distrib = D.MultivariateNormal(
            (mat_Cs[:-1] @ zs_sample.unsqueeze(-1)).view(-1, self.a_dim),
            self.state_space_model.mat_R,
        )
        kalman_observation_log_likelihood = aggregate(
            kalman_observation_distrib.log_prob(as_.view(-1, self.a_dim)).view(
                seq_length, batch_size
            ),
            sequence_length=seq_length,
            batch_size=batch_size,
            sequence_operation=self.config.sequence_operation,
            batch_operation=self.config.batch_operation,
        )

        # ln p_\gamma(z) = \ln p_\gamma(z_0) + \sum_{t=1}^{T-1} ln p_\gamma(z_t|z_{t-1})
        zs_prior_means = torch.cat(
            [
                self.state_space_model.initial_state_mean.repeat(1, batch_size, 1),
                (mat_As[1:-1] @ zs_sample[:-1].unsqueeze(-1)).squeeze(-1),
            ]
        )
        zs_prior_covariances = torch.cat(
            [
                self.state_space_model.initial_state_covariance.repeat(
                    1, batch_size, 1, 1
                ),
                self.state_space_model.mat_Q.repeat(seq_length - 1, batch_size, 1, 1),
            ]
        )
        zs_prior_distrib = D.MultivariateNormal(
            zs_prior_means.view(seq_length, batch_size, self.z_dim),
            zs_prior_covariances.view(seq_length, batch_size, self.z_dim, self.z_dim),
        )
        kalman_state_transition_log_likelihood = aggregate(
            zs_prior_distrib.log_prob(zs_sample),
            sequence_length=seq_length,
            batch_size=batch_size,
            sequence_operation=self.config.sequence_operation,
            batch_operation=self.config.batch_operation,
        )

        # ln p_\gamma(z|a)
        kalman_posterior_log_likelihood = aggregate(
            zs_distrib.log_prob(zs_sample).view(seq_length, batch_size),
            sequence_length=seq_length,
            batch_size=batch_size,
            sequence_operation=self.config.sequence_operation,
            batch_operation=self.config.batch_operation,
        )

        weighted_kl_reg = self.config.kl_weight * kl_reg
        weighted_kalman_observation_log_likelihood = (
            self.config.kalman_weight * kalman_observation_log_likelihood
        )
        weighted_kalman_state_transition_log_likelihood = (
            self.config.kalman_weight * kalman_state_transition_log_likelihood
        )
        weighted_kalman_posterior_log_likelihood = (
            -self.config.kalman_weight * kalman_posterior_log_likelihood
        )

        objective = (
            +weighted_kl_reg
            + weighted_kalman_observation_log_likelihood
            + weighted_kalman_state_transition_log_likelihood
            + weighted_kalman_posterior_log_likelihood
        )

        if xs is not None:
            weighted_reconstruction_obj = (
                self.config.reconst_weight * reconstruction_obj
            )
            weighted_regularization_obj = (
                -self.config.regularization_weight * regularization_obj
            )
            objective += weighted_reconstruction_obj + weighted_regularization_obj

        return objective, {
            "reconst_weight": self.config.reconst_weight,
            "regularization_weight": self.config.regularization_weight,
            "kalman_weight": self.config.kalman_weight,
            "kl_weight": self.config.kl_weight,
            "reconstruction": weighted_reconstruction_obj if xs is not None else 0.0,
            "regularization": weighted_regularization_obj if xs is not None else 0.0,
            "kl": weighted_kl_reg,
            "kalman_observation_log_likelihood": weighted_kalman_observation_log_likelihood,
            "kalman_state_transition_log_likelihood": weighted_kalman_state_transition_log_likelihood,
            "kalman_posterior_log_likelihood": weighted_kalman_posterior_log_likelihood,
            "observation_mask": observation_mask,
            "filter_means": filter_means,
            "filter_covariances": filter_covariances,
            "filter_next_means": filter_next_means,
            "filter_next_covariances": filter_next_covariances,
            "mat_As": mat_As,
            "mat_Cs": mat_Cs,
            "weights": weights,
            "means": means,
            "covariances": covariances,
            "as": as_,
            "zs": zs,
            "filter_as": filter_as,
            "as_resampled": as_resampled,
        }
