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

    def encoder(self, x):
        """Implement the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, x_dim).
        Returns
        -------
        tuple
            A tuple containing the mean and log variance of the latent variable.
        """
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        """Sample latent variables from the Gaussian distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the latent variable.
        log_var : torch.Tensor
            Log variance of the latent variable.
        device : torch.device
            Device to perform the computation on.
        Returns
        -------
        torch.Tensor
            Sampled latent variable z.
        """
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z):
        """
        Implement the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent variable tensor of shape (batch_size, z_dim).
        Returns
        -------
        torch.Tensor
            Reconstructed output tensor of shape (batch_size, x_dim).
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        """
        Implement the forward function to return the following variables.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, x_dim).
        device : torch.device
            Device to perform the computation on.
        Returns
        -------
        list
            A list containing the KL divergence and reconstruction loss.
        torch.Tensor
            Sampled latent variable z.
        torch.Tensor
            Reconstructed output tensor y.
        """
        mean, log_var = self.encoder(x.to(device))
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )
        return [KL, reconstruction], z, y
