"""Train the diffusion model for image denoising."""

import logging
import os
from typing import Any, Dict

import diffusers
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import imageio
from torchvision.utils import make_grid


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
        a = self.alpha_prod[t].reshape(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse process of the diffusion model. x_t ~ p(x_t|x_{t+1}).

        Args:
            x (torch.Tensor): Noisy image x_{t+1} (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Denoised image x_t (B, C, H, W)
        """
        with torch.no_grad():
            noise_pred = self.model(x, t)

            alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_prod[t].reshape(-1, 1, 1, 1)
            beta_t = self.beta[t].reshape(-1, 1, 1, 1)

            mu = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * noise_pred.sample)

            noise = torch.randn_like(x)
            nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)

            return mu + nonzero_mask * beta_t.sqrt() * noise

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

        t = torch.randint(0, self.num_timesteps, (images.size(0),), device=self.device)

        noise = torch.randn_like(images)
        x_t = self.q_sample(images, t, noise)

        predict_noise = self.forward(x_t, t)

        loss = self.criterion(predict_noise, noise)

        return loss

    def generate_with_gif(self, num_timesteps: int, shape: tuple, gif_path: str, final_image_path: str = None):
        """generate gif and final image

        Args:
            num_timesteps (int): time steps
            shape (tuple): image shape (B, C, H, W)
            gif_path (str): path to gif
            final_image_path (str): path to final image
        """
        x = torch.randn(shape).to(self.device)
        frames = []

        for t_step in tqdm(range(num_timesteps - 1, -1, -1), desc="Generating with GIF"):
            t = torch.full((x.size(0),), t_step, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)

            # 将 x 映射回 [0, 1] 再转 uint8 图像帧
            frame = (x.clone().detach().cpu() + 1) / 2  # [0,1]
            frame_grid = make_grid(frame, nrow=self.num_samples[0])
            frame_np = (frame_grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
            frames.append(frame_np)

        # save gif
        imageio.mimsave(gif_path, frames, fps=10)

        # save final image
        if final_image_path:
            from PIL import Image
            final_image = Image.fromarray(frames[-1])
            final_image.save(final_image_path)

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


@hydra.main(config_path="../conf", config_name="default.yaml", version_base=None)
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

    # call after training
    gif_path = os.path.join(outdir, "generation.gif")
    final_image_path = os.path.join(outdir, "final_image.png")
    diffmodel.generate_with_gif(
        num_timesteps=cfg.diffusion.num_timesteps,
        shape=(cfg.plot.num_samples[0] * cfg.plot.num_samples[1],) + tuple(cfg.plot.image_size),
        gif_path=gif_path,
        final_image_path=final_image_path
    )

    # save model
    torch.save(diffmodel.state_dict(), outdir + "/model.pth")


if __name__ == "__main__":
    main()
