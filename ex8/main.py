# -*- coding: utf-8 -*-
"""This file is for you to implement the main function."""

import os

import fire
import numpy as np
import torch
from libs.Visualize import Visualize
from torch import optim
from torchvision import datasets, transforms
from VAEs.VAE import VAE


class Main:
    """Main class for training and visualization."""

    def __init__(
        self,
        z_dim: int = 2,
        h_dim: int = 400,
        drop_rate: float = 0.2,
        learning_rate: float = 0.001,
        num_max_epochs: int = 1000,
        do_train: bool = True,
        train_size_rate: float = 0.8,
    ):
        """
        Set constructors.

        Parameters
        ----------
        z_dim : int
            Dimensions of the latent variable, by default 2.
            Attention: If you visualize the latent space with a dimension greater than 2,
            you need to change the code in libs/Visualize.py.
        h_dim : int, optional
            Dimensions of the hidden layer, by default 400.
        drop_rate : float, optional
            Dropout rate, by default 0.2.
        learning_rate : float, optional
            Learning rate, by default 0.001.
        num_max_epochs : int, optional
            The number of epochs for training, by default 1000.
        do_train : bool, optional
            Whether to train the model, by default True.
        train_size_rate : float, optional
            The ratio of the training data to the validation data, by default 0.8.
        """
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.drop_rate = drop_rate
        self.lr = learning_rate
        self.num_max_epochs = num_max_epochs
        self.do_train = do_train
        self.train_size_rate = train_size_rate
        self.batch_size = 625

        self.dataloader_train = None
        self.dataloader_valid = None
        self.dataloader_test = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(self.z_dim, self.h_dim, self.drop_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_no_improved = 0
        self.num_batch_train = 0
        self.num_batch_valid = 0
        self.loss_valid = 10**7  # Initialize with a large value
        self.loss_valid_min = 10**7  # Initialize with a large value
        self.Visualize = Visualize(
            self.z_dim, self.h_dim, self.dataloader_test, self.model, self.device
        )

    def createDirectories(self):
        """Create directories for logs and parameters."""
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./params", exist_ok=True)

    def createDataLoader(self):
        """Create dataloaders for training, validation, and test data."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )  # Preprocessing for MNIST images
        dataset_train_valid = datasets.MNIST(
            "./", train=True, download=True, transform=transform
        )  # Separate train data and test data to get a dataset
        dataset_test = datasets.MNIST(
            "./", train=False, download=True, transform=transform
        )

        # Separate the training data and validation data
        size_train_valid = len(dataset_train_valid)
        size_train = int(size_train_valid * self.train_size_rate)
        size_valid = size_train_valid - size_train
        dataset_train, dataset_valid = torch.utils.data.random_split(
            dataset_train_valid, [size_train, size_valid]
        )

        # Create dataloaders from the datasets
        self.dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True
        )
        self.dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=self.batch_size, shuffle=False
        )
        self.dataloader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.batch_size, shuffle=False
        )
        self.Visualize.dataloader_test = self.dataloader_test

    def train_batch(self):
        """Do batch-based learning for training data."""
        self.model.train()
        for x, _ in self.dataloader_train:
            lower_bound, _, _ = self.model(x, self.device)
            loss = -sum(lower_bound)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_batch_train += 1
        self.num_batch_train -= 1

    def valid_batch(self):
        """Do batch-based learning for validation data."""
        loss = []
        self.model.eval()
        for x, _ in self.dataloader_valid:
            lower_bound, _, _ = self.model(x, self.device)
            loss.append(-sum(lower_bound).cpu().detach().numpy())
            self.num_batch_valid += 1
        self.num_batch_valid -= 1
        self.loss_valid = np.mean(loss)
        self.loss_valid_min = np.minimum(self.loss_valid_min, self.loss_valid)

    def early_stopping(self):
        """Set a condition for early stopping."""
        if self.loss_valid_min < self.loss_valid:
            # If the loss of this iteration is greater than the minimum loss of
            # the previous iterations, the counter variable is incremented.
            self.num_no_improved += 1
            print(f"Validation got worse for the {self.num_no_improved} time in a row.")
        else:
            # If the loss of this iteration is the same or smaller than the minimum loss of
            # the previous iterations, reset the counter variable and save parameters.
            self.num_no_improved = 0
            torch.save(
                self.model.state_dict(),
                f"./params/model_z_{self.z_dim}_h_{self.h_dim}.pth",
            )

    def main(self):
        """Output the results of training and visualization."""
        self.createDirectories()
        self.createDataLoader()

        if self.do_train:
            # print("-----Start training-----")
            for self.num_iter in range(self.num_max_epochs):
                self.train_batch()
                self.valid_batch()
                print(
                    f"[EPOCH{self.num_iter + 1}] loss_valid: {int(self.loss_valid)}"
                    + f" | Loss_valid_min: {int(self.loss_valid_min)}"
                )
                self.early_stopping()
                if self.num_no_improved >= 10:
                    # print("Apply early stopping")
                    break
            # print("-----Stop training-----")

        # print("-----Start Visualization-----")
        self.model.load_state_dict(
            torch.load(f"./params/model_z_{self.z_dim}_h_{self.h_dim}.pth")
        )
        self.model.eval()
        self.Visualize.createDirectories()
        self.Visualize.reconstruction()
        self.Visualize.latent_space()
        self.Visualize.lattice_point()
        self.Visualize.walkthrough()
        # print("-----Stop Visualization-----")


if __name__ == "__main__":
    fire.Fire(Main)
