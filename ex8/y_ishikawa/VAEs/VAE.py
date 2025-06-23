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
        self.eps = np.spacing(1)
        self.x_dim = MNIST_SIZE * MNIST_SIZE  # The image in MNIST is 28×28
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.drop_rate = drop_rate

        self.enc_fc1 = nn.Linear(self.x_dim, self.h_dim)
        self.enc_fc2 = nn.Linear(self.h_dim, int(self.h_dim / 2))
        self.enc_fc3_mean = nn.Linear(int(self.h_dim / 2), z_dim)
        self.enc_fc3_logvar = nn.Linear(int(self.h_dim / 2), z_dim)
        self.dec_fc1 = nn.Linear(z_dim, int(self.h_dim / 2))
        self.dec_fc2 = nn.Linear(int(self.h_dim / 2), self.h_dim)
        self.dec_drop = nn.Dropout(self.drop_rate)
        self.dec_fc3 = nn.Linear(self.h_dim, self.x_dim)

    def encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, MNIST_SIZE * MNIST_SIZE).

        Returns
        -------
        mean : torch.Tensor
            Mean of the latent variable, shape (batch_size, z_dim).
        logvar : torch.Tensor
            Log variance of the latent variable, shape (batch_size, z_dim).
        """
        x = x.view(-1, self.x_dim)
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        mean = self.enc_fc3_mean(x)
        logvar = self.enc_fc3_logvar(x)
        return mean, logvar

    def sample_z(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample the latent variable z from the Gaussian distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the latent variable, shape (batch_size, z_dim).
        logvar : torch.Tensor
            Log variance of the latent variable, shape (batch_size, z_dim).

        Returns
        -------
        z : torch.Tensor
            Sampled latent variable z, shape (batch_size, z_dim).
        """
        z = mean + torch.exp(logvar / 2) * torch.randn_like(mean)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent variable z to reconstruct the input data.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable z, shape (batch_size, z_dim).

        Returns
        -------
        y : torch.Tensor
            Reconstructed input data, shape (batch_size, MNIST_SIZE * MNIST_SIZE).
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        y = torch.sigmoid(self.dec_fc3(z))
        return y

    def forward(
        self, x: torch.Tensor, device: torch.device
    ) -> tuple[list, torch.Tensor, torch.Tensor]:
        """Return KL divergence, reconstruction loss, latent variable, and output data.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, MNIST_SIZE * MNIST_SIZE).
        device : torch.device
            Device to run the model on, either "mps" (Apple Silicon), "cuda" (NVIDIA GPU), or "cpu".

        Returns
        -------
        list
            List containing KL divergence and reconstruction loss.
        z : torch.Tensor
            Latent variable, shape (batch_size, z_dim).
        y : torch.Tensor
            Output data, shape (batch_size, MNIST_SIZE * MNIST_SIZE).
        """
        x = x.to(device)

        mean, logvar = self.encoder(x)
        z = self.sample_z(mean, logvar)
        y = self.decoder(z)

        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        KL = torch.sum(1 + logvar - mean**2 - torch.exp(logvar)) / 2

        return [KL, reconstruction], z, y
