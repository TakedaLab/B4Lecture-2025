# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_SIZE = 28


class VAE(nn.Module):
    """VAE model."""

    def __init__(self, z_dim, h_dim, drop_rate):
        """Set constructors.

        Parameters
        ----------
        z_dim : int
            Dimensions of the latent variable.
        h_dim : int
            Dimensions of the hidden layer.
        drop_rate : float
            Dropout rate.
        """
        super(VAE, self).__init__()
        self.eps = np.spacing(1)  # Tiny number epsilon
        self.x_dim = MNIST_SIZE * MNIST_SIZE  # The image in MNIST is

        # Constructor parameters
        self.z_dim = z_dim
        self.h_dim = h_dim  # Default 400
        self.drop_rate = drop_rate

        # Model layers

        # Encoding
        self.enc_fc1 = nn.Linear(self.x_dim, self.h_dim)
        self.enc_fc2 = nn.Linear(self.h_dim, int(self.h_dim / 2))
        self.enc_fc3_mean = nn.Linear(int(self.h_dim / 2), z_dim)
        self.enc_fc3_logvar = nn.Linear(int(self.h_dim / 2), z_dim)

        # Decoding
        self.dec_fc1 = nn.Linear(z_dim, int(self.h_dim / 2))
        self.dec_fc2 = nn.Linear(int(self.h_dim / 2), self.h_dim)
        self.dec_drop = nn.Dropout(self.drop_rate)
        self.dec_fc3 = nn.Linear(self.h_dim, self.x_dim)

    def encoder(self, x):
        """Run data through the encoder model.

        Args:
            x (torch.Tensor): Input data tensor.
        Returns:
            (torch.Tensor, torch.Tensor): The mean and log variance tensors
        """
        # Apply encoding layers with ReLU
        x = self.enc_fc1(x)
        x = F.relu(x)

        x = self.enc_fc2(x)
        x = F.relu(x)

        # Compress into latent space
        mean = self.enc_fc3_mean(x)
        log_var = self.enc_fc3_logvar(x)

        return mean, log_var

    def sample_z(self, mean: torch.Tensor, log_var: torch.Tensor, device: torch.device):
        """Retrieve sample from latent space.

        Args:
            mean (torch.Tensor):
            log_var (torch.Tensor):
        Returns:
            torch.Tensor: The generated sample.
        """
        input_shape = mean.shape  # (batch, dim)

        # Sample from standard normal distribution
        sample_norm = torch.randn(input_shape, device=device)

        # Fit sample to be of the input distribution
        return mean + torch.exp(log_var * 0.5) * sample_norm

    def decoder(self, z):
        """Reconstruct data from the latent sample.

        Args:
            z (torch.Tensor): The retrieved sample of the latent distribution.
        Returns:
            torch.Tensor: The decoded data tensor.
        """
        # Apply decoding layers with ReLU
        z = self.dec_fc1(z)
        z = F.relu(z)

        z = self.dec_fc2(z)
        z = F.relu(z)

        z = self.dec_drop(z)

        # Use sigmoid for output layer
        z = self.dec_fc3(z)
        z = F.sigmoid(z)

        return z

    def forward(self, x, device):
        """Perform forward pass and return data and loss metrics.

        Args:
            x (torch.Tensor): Input data tensor
        Returns:
            list: [KL divergence, reconstruction loss]
            torch.Tensor: Latent variable
            torch.Tensor: Output data
        """
        # Use gpu if available
        x = x.to(device)

        # Encode
        mean, log_var = self.encoder(x)

        # Sample
        z = self.sample_z(mean, log_var, device)

        # Decode
        y = self.decoder(z)

        # Calculate KL divergence loss
        KL = 0.5 * torch.sum(1 + log_var - torch.square(mean) - torch.exp(log_var))

        # Calculate reconstruction loss (eps to avoid log(0))
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )

        return [KL, reconstruction], z, y
