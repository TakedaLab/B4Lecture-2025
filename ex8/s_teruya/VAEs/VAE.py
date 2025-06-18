# -*- coding: utf-8 -*-
"""This file is for you to implement VAE. Add variables as needed."""

import numpy as np
import torch
import torch.nn as nn

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
        self.relu=nn.ReLU()
        self.rec_method=nn.BCELoss(reduction="sum")

    def encoder(self, x):
        """# ToDo: Implement the encoder."""
        x=x.reshape((-1,self.x_dim))    # https://qiita.com/kenta1984/items/d68b72214ce92beebbe2
        x=self.relu(self.enc_fc1(x))
        x=self.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, enc_mean, enc_logvar, device):
        """# ToDo: Implement a function to sample latent variables."""
        enc_std=torch.exp(enc_logvar / 2)   # https://qiita.com/nabenabe0928/items/342fb7829abe31d1bd49
        ns=torch.randn(enc_std.shape).to(device)    # https://tech.aru-zakki.com/python-random/
        return enc_mean+ns*enc_std

    def decoder(self, z):
        """# ToDo: Implement the decoder."""
        z=self.relu(self.dec_fc1(z))
        z=self.relu(self.dec_fc2(z))
        z=self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        """# ToDo: Implement the forward function to return the following variables."""
        x = x.to(device)
        enc_mean, enc_logvar=self.encoder(x)
        z=self.sample_z(enc_mean, enc_logvar, device)
        y=self.decoder(z)
        
        KL=torch.sum(1+enc_logvar-enc_mean**2-torch.exp(enc_logvar))/2  # https://qiita.com/gensal/items/613d04b5ff50b6413aa0
        reconstruction=-self.rec_method(y, x)   # https://qiita.com/PingpongChopper/items/d7db77516c52b9bb15c6
        return [KL, reconstruction], z, y
