import logging
import os
from typing import Any, Dict

import diffusers
import hydra
import imageio
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class ImageTransform:
    def __init__(self, image_size, to_rgb: bool = True):
        self.to_rgb = to_rgb
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, examples):
        if self.to_rgb:
            images = [
                self.transform(image.convert("RGB")) for image in examples["image"]
            ]
        else:
            images = [self.transform(image) for image in examples["image"]]
        return {"images": images}


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
        self.save_hyperparameters(ignore=["model", "criterion", "optimizer"])
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch_images = []

        if self.hparams.noise_schedule == "linear":
            beta = torch.linspace(
                self.hparams.noise_schedule_kwargs["start"],
                self.hparams.noise_schedule_kwargs["end"],
                self.hparams.num_timesteps,
            )
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)
            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.optimizer.defaults["lr"]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

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
        a = self.alpha_prod[t].view(-1, 1, 1, 1)
        return x0 * a.sqrt() + noise * (1 - a).sqrt()

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inverse process of the diffusion model. x_t ~ p(x_t|x_{t+1}).

        Args:
            x (torch.Tensor): Noisy image x_{t+1} (B, C, H, W)
            t (torch.Tensor): Time step (B,)

        Returns:
            torch.Tensor: Denoised image x_t (B, C, H, W)
        """
        with torch.no_grad():
            noise_pred = self.forward(x, t)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
            x = (
                x - noise_pred * (1 - alpha_t) / (1 - alpha_prod_t).sqrt()
            ) / alpha_t.sqrt()
            if t[0].item() != 0:
                x = x + torch.randn_like(x) * (1 - alpha_t).sqrt()
            return x

    def training_step(self, batch, batch_idx):
        """Training 1 step.

        Args:
            batch (tuple): Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss
        """
        images = batch["images"].to(self.device)

        t = torch.randint(
            0, self.hparams.num_timesteps, (images.size(0),), device=self.device
        ).long()

        noise = torch.randn_like(images)
        noisy_images = self.q_sample(images, t, noise)

        predicted_noise = self.forward(noisy_images, t)
        loss = self.criterion(predicted_noise, noise)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def generate(self, num_timesteps: int, shape: tuple) -> torch.Tensor:
        """Generate samples from the diffusion model."""
        x = torch.randn(shape, device=self.device)
        for t in tqdm(range(num_timesteps - 1, -1, -1), desc="Generating image"):
            t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t_tensor)
        return x

    def on_train_epoch_end(self):
        """Generate images at the end of each epoch."""
        if (
            self.current_epoch % self.hparams.every_n_epochs
            == self.hparams.every_n_epochs - 1
        ):
            logging.info(f"Generating images at epoch {self.current_epoch}...")

            shape = (
                self.hparams.num_samples[0] * self.hparams.num_samples[1],
            ) + tuple(self.hparams.image_size)
            generated_image = self.generate(self.hparams.num_timesteps, shape)
            generated_image = (generated_image + 1) / 2
            grid_image = (
                generated_image.reshape(
                    tuple(self.hparams.num_samples) + tuple(self.hparams.image_size)
                )
                .permute([2, 0, 3, 1, 4])
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
            self.logger.experiment.add_image(
                "Generated Images",
                grid_image,
                self.current_epoch,
            )

            grid_image_np = (grid_image.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            if grid_image_np.shape[0] == 1:
                self.epoch_images.append(grid_image_np.squeeze(0))
            else:
                self.epoch_images.append(grid_image_np.transpose(1, 2, 0))

            logging.info("Done.")

    def on_train_end(self):
        """
        PyTorch Lightning hook called at the very end of training.
        Used here to generate final GIF animations.
        """
        logging.info("Training finished. Generating GIFs...")
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # Create a GIF of the model's progress over epochs.
        if self.epoch_images:
            gif_path = os.path.join(output_dir, "model_progress.gif")
            imageio.mimsave(gif_path, self.epoch_images, fps=2)
            logging.info(f"Model progress GIF saved to {gif_path}")

        # Create a GIF of a single reverse diffusion process.
        diffusion_process_images = []
        shape = (1,) + tuple(self.hparams.image_size)
        x = torch.randn(shape, device=self.device)
        for t in tqdm(
            range(self.hparams.num_timesteps - 1, -1, -1),
            desc="Generating diffusion process GIF",
        ):
            if t % 50 == 0:
                img_t = (x + 1) / 2
                img_np = (img_t.clamp(0, 1).squeeze(0).cpu().numpy() * 255).astype(
                    "uint8"
                )
                diffusion_process_images.append(img_np.transpose(1, 2, 0))

            t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t_tensor)

        img_final = (x + 1) / 2
        img_np_final = (img_final.clamp(0, 1).squeeze(0).cpu().numpy() * 255).astype(
            "uint8"
        )
        diffusion_process_images.append(img_np_final.transpose(1, 2, 0))
        gif_path_diffusion = os.path.join(output_dir, "diffusion_process.gif")
        imageio.mimsave(gif_path_diffusion, diffusion_process_images, fps=10)
        logging.info(f"Diffusion process GIF saved to {gif_path_diffusion}")


@hydra.main(config_path="conf", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    torch.set_float32_matmul_precision("medium")

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    train_dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=cfg.datadir
    )
    # preprocess = transforms.Compose(
    #     [
    #         transforms.Resize(cfg.plot.image_size[-2:]),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    # def transform(examples):
    #     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    #     return {"images": images}

    transform = ImageTransform(tuple(cfg.plot.image_size[-2:]))

    train_dataset.set_transform(transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
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

    tb_logger = TensorBoardLogger(outdir, name="diffusion_logs")
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="train_loss", mode="min")],
    )
    trainer.fit(diffmodel, train_loader)

    torch.save(diffmodel.state_dict(), os.path.join(outdir, "model.pth"))


if __name__ == "__main__":
    main()
