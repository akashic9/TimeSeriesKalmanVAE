import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from model.misc import compute_conv2d_output_size


class ImageEncoder(nn.Module):
    def __init__(self, image_size, image_channels, a_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LayerNorm([32, image_size[0] // 2, image_size[1] // 2]),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LayerNorm([64, image_size[0] // 4, image_size[1] // 4]),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.LayerNorm([128, image_size[0] // 8, image_size[1] // 8]),
            nn.LeakyReLU(),
        )

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )

        self.fc_mean = nn.Linear(
            in_features=128 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )
        self.fc_std = nn.Linear(
            in_features=128 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
        nn.init.kaiming_normal_(
            self.fc_mean.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.fc_std.weight, mode="fan_out", nonlinearity="leaky_relu"
        )

    def forward(self, x):
        x = self.conv_layers(x).view(x.shape[0], -1)
        x_mean = self.fc_mean(x)
        x_std = F.softplus(self.fc_std(x))
        return D.Normal(loc=x_mean, scale=x_std)


class ImageDecoder(nn.Module):
    def __init__(self, image_size, image_channels, a_dim):
        super().__init__()

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )
        self.conv_output_size = conv_output_size

        self.fc = nn.Linear(
            in_features=a_dim,
            out_features=128 * conv_output_size[0] * conv_output_size[1],
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LayerNorm([64, conv_output_size[0] * 2, conv_output_size[1] * 2]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.LayerNorm([32, conv_output_size[0] * 4, conv_output_size[1] * 4]),
            nn.LeakyReLU(),
        )

        self.deconv3_mean = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3_std = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.deconv_layers:
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
        nn.init.kaiming_normal_(
            self.fc.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.deconv3_mean.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.kaiming_normal_(
            self.deconv3_std.weight, mode="fan_out", nonlinearity="leaky_relu"
        )

    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        x = x.view(-1, 128, *self.conv_output_size)
        x = self.deconv_layers(x)
        x_mean = self.deconv3_mean(x)
        x_std = F.softplus(self.deconv3_std(x))
        return D.Normal(loc=x_mean, scale=x_std)


class IndependentEncoder(nn.Module):
    def __init__(self, image_size, a_dim):
        super().__init__()
        self.encoder1 = ImageEncoder(image_size, 1, a_dim // 2)
        self.encoder2 = ImageEncoder(image_size, 1, a_dim // 2)
        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )

        self.fc_mean = nn.Linear(
            in_features=2 * 128 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )
        self.fc_std = nn.Linear(
            in_features=2 * 128 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )

    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        x1_features = self.encoder1.conv_layers(x1).view(x1.shape[0], -1)
        x2_features = self.encoder2.conv_layers(x2).view(x2.shape[0], -1)
        combined_features = torch.cat([x1_features, x2_features], dim=-1)
        x_mean = self.fc_mean(combined_features)
        x_std = F.softplus(self.fc_std(combined_features))
        return D.Normal(loc=x_mean, scale=x_std)


class IndependentDecoder(nn.Module):
    def __init__(self, image_size, a_dim):
        super().__init__()
        self.a_dim = a_dim
        self.decoder1 = ImageDecoder(image_size, 1, a_dim // 2)
        self.decoder2 = ImageDecoder(image_size, 1, a_dim // 2)

    def forward(self, x):
        x1 = x[:, : self.a_dim // 2]
        x1 = F.leaky_relu(self.decoder1.fc(x1))
        x1 = x1.view(-1, 128, *self.decoder1.conv_output_size)
        x1 = self.decoder1.deconv_layers(x1)
        x1_mean = self.decoder1.deconv3_mean(x1)
        x1_std = F.softplus(self.decoder1.deconv3_std(x1))

        x2 = x[:, self.a_dim // 2 :]
        x2 = F.leaky_relu(self.decoder2.fc(x2))
        x2 = x2.view(-1, 128, *self.decoder2.conv_output_size)
        x2 = self.decoder2.deconv_layers(x2)
        x2_mean = self.decoder2.deconv3_mean(x2)
        x2_std = F.softplus(self.decoder2.deconv3_std(x2))

        x_mean = torch.cat([x1_mean, x2_mean], dim=1)
        x_std = torch.cat([x1_std, x2_std], dim=1)
        return D.Normal(loc=x_mean, scale=x_std)
