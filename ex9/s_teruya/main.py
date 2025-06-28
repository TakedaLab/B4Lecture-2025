"""Train the diffusion model for image denoising."""

import logging
import os
import traceback
from typing import Any, Dict

import diffusers
import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from matplotlib.animation import ArtistAnimation
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class DiffusionModel(pl.LightningModule):
    """Diffusion model for image denoising."""

    def __init__(
        self,
        model: torch.nn.Module,  # Noise prediction model
        criterion: torch.nn.Module,  # Loss function
        optimizer: torch.optim.Optimizer,  # Optimizer
        num_timesteps: int,  # Time steps of the diffusion
        noise_schedule: str,  # Noise scheduler type
        noise_schedule_kwargs: Dict[str, Any],  # Arguments for noise scheduler
        num_samples: tuple,  # Number of samples for visualization
        image_size: tuple,  # Image size
        every_n_epochs: int,  # Visualization interval
    ) -> None:
        """Initialize the diffusion model."""
        super(DiffusionModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.num_timesteps = num_timesteps

        if noise_schedule == "linear":
            beta = torch.linspace(
                noise_schedule_kwargs["start"],
                noise_schedule_kwargs["end"],
                num_timesteps,
                device=model.device,
            )
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)
            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

        self.num_samples = tuple(num_samples)
        self.image_size = tuple(image_size)
        self.every_n_epochs = every_n_epochs

    def configure_optimizers(self):
        """Configure optimizer."""
        return self.optimizer

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward noise prediction model.

        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Predicted noise (B, C, H, W)
        """
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward process of the diffusion model. x_t ~ q(x_t|x_0).

        Args:
            x0 (torch.Tensor): Clean image x_0 (B, C, H, W)
            t (torch.Tensor): Time step (B,)
            noise (torch.Tensor, optional): Noise tensor. Defaults to None.

        Returns:
            torch.Tensor: Noisy image x_t (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_prod_t) * x0 + torch.sqrt(1 - alpha_prod_t) * noise
        return x_t

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse process of the diffusion model. x_t ~ p(x_t|x_{t+1}).

        Args:
            x (torch.Tensor): Noisy image x_{t+1} (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Denoised image x_t (B, C, H, W)
        """
        with torch.no_grad():
            noise = self.forward(x, t)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)

            x_t = (
                x - noise * (1 - alpha_t) / torch.sqrt(1 - alpha_prod_t)
            ) / torch.sqrt(alpha_t)
            if t[0] > 0:
                x_t += torch.sqrt(beta_t) * torch.randn_like(x)
            return x_t

    def training_step(self, batch, batch_idx):
        """Training 1 step.

        Args:
            batch (tuple): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss
        """
        images = batch["images"]
        images = images.to(self.device)

        t = torch.randint(0, self.num_timesteps, (images.shape[0],), device=self.device)
        noise = torch.randn_like(images)
        noisy_images = self.q_sample(images, t, noise)

        pred_noise = self.forward(noisy_images, t)
        loss = self.criterion(noise, pred_noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def generate(self, num_timesteps, shape):
        """Generate samples from the diffusion model."""
        x = torch.randn(shape).to(self.device)
        for t in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
        return x

    def on_train_epoch_end(self):
        """Generate images at the end of each epoch."""
        if self.current_epoch % self.every_n_epochs == self.every_n_epochs - 1:
            logging.info("Generating Images...")
            generated_image = self.generate(
                self.num_timesteps,
                (self.num_samples[0] * self.num_samples[1],) + self.image_size,
            )
            generated_image = (generated_image + 1) / 2
            generated_image = (
                generated_image.reshape(self.num_samples + self.image_size)
                .permute([2, 0, 3, 1, 4])
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
            self.logger.experiment.add_image(
                "Generated Images",
                generated_image,
                self.current_epoch,
                dataformats="HW" if generated_image.ndim == 2 else "CHW",
            )
            logging.info("Done.")

    def on_train_end(self):
        """Generate gifs at the end of train."""
        print("generating gifs...")
        x = torch.randn((4,) + self.image_size).to(self.device)
        image_list = [torch.clamp((x + 1) / 2, 0.0, 1.0).cpu().detach().numpy()]
        for t in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
            if t % 10 == 0:
                image_list.append(
                    torch.clamp((x + 1) / 2, 0.0, 1.0).cpu().detach().numpy()
                )
        try:
            os.makedirs(f"{self.logger.log_dir}/gifs", exist_ok=True)
            for n in range(image_list[0].shape[0]):
                fig, ax = plt.subplots(1, 1, figsize=(9, 9))
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax.set_xticks([])
                ax.set_yticks([])
                images = []
                for _, im in enumerate(image_list):
                    images.append([ax.imshow(im[n].transpose(1, 2, 0), vmin=0, vmax=1)])
                animation = ArtistAnimation(
                    fig, images, interval=100, blit=True, repeat_delay=1000
                )
                animation.save(
                    f"{self.logger.log_dir}/gifs/generate_{n}.gif", writer="pillow"
                )
                plt.close(fig)
        except Exception:
            print("Warning!: Failed to generate gifs.")
            traceback.print_exc()


@hydra.main(config_path="conf", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train the diffusion model."""
    # fix seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # prepare dataset
    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=cfg.datadir
    )
    # preprocess
    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.plot.image_size[-2:]),  # Resize
            transforms.ToTensor(),  # ToTensor
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ]
    )

    def transform(examples):
        """Transform the dataset."""
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    model = diffusers.UNet2DModel(**cfg.model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)

    diffmodel = DiffusionModel(
        model,
        criterion,
        optimizer,
        **cfg.diffusion,
        **cfg.plot,
    )

    diffmodel.save_hyperparameters(cfg)

    # configure logger
    tb_logger = TensorBoardLogger(outdir)
    # train
    trainer = pl.Trainer(max_epochs=cfg.train.num_epochs, devices=1, logger=tb_logger)
    trainer.fit(diffmodel, train_loader)

    # save model
    torch.save(diffmodel.state_dict(), outdir + "/model.pth")


if __name__ == "__main__":
    main()
