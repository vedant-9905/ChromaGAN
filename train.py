"""
train.py
========
Production Training Pipeline for ChromaGAN Conditional Generative Adversarial Network.
Trains U-Net Generator and PatchGAN Discriminator in CIELAB color space.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.dataset import get_colorization_dataloader
from core.losses import ChromaGANLoss
from core.models import PatchGANColorDiscriminator, UNetColorGenerator


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[ChromaGAN Trainer] Starting training on device: {device}")

    # Create checkpoint directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = save_dir / "visual_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Dataloader
    print(f"[ChromaGAN Trainer] Loading data from {args.data_dir}...")
    dataloader = get_colorization_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        is_train=True,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"[ChromaGAN Trainer] Total training batches per epoch: {len(dataloader)}")

    # Models
    generator = UNetColorGenerator(in_channels=1, out_channels=2, base_filters=64).to(device)
    discriminator = PatchGANColorDiscriminator(in_channels=3, base_filters=64).to(device)

    # Losses
    criterion = ChromaGANLoss(lambda_l1=args.lambda_l1).to(device)

    # Optimizers (Standard DCGAN / Pix2Pix settings: lr=2e-4, beta1=0.5, beta2=0.999)
    optimizer_g = Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    scheduler_g = StepLR(optimizer_g, step_size=args.decay_step, gamma=0.5)
    scheduler_d = StepLR(optimizer_d, step_size=args.decay_step, gamma=0.5)

    start_epoch = 1

    # Optional resume
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[ChromaGAN Trainer] Resuming checkpoint from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device)
            generator.load_state_dict(ckpt["generator"])
            discriminator.load_state_dict(ckpt["discriminator"])
            optimizer_g.load_state_dict(ckpt["opt_g"])
            optimizer_d.load_state_dict(ckpt["opt_d"])
            start_epoch = ckpt.get("epoch", 0) + 1

    print("[ChromaGAN Trainer] Ready. Beginning adversarial optimization loop.")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_l1_loss = 0.0

        generator.train()
        discriminator.train()

        for step, (l_batch, ab_batch, rgb_batch) in enumerate(dataloader):
            l_batch = l_batch.to(device)      # [B, 1, H, W]
            ab_batch = ab_batch.to(device)    # [B, 2, H, W]

            # ---------------------------------------------------------------
            # 1. Train Discriminator: Maximize log(D(x, y)) + log(1 - D(x, G(x)))
            # ---------------------------------------------------------------
            optimizer_d.zero_grad()

            # Forward generator
            ab_fake = generator(l_batch)

            # Discriminator real: [L, ab_real]
            real_pair = torch.cat([l_batch, ab_batch], dim=1)
            d_real_preds = discriminator(real_pair)

            # Discriminator fake: [L, ab_fake.detach()]
            fake_pair = torch.cat([l_batch, ab_fake.detach()], dim=1)
            d_fake_preds = discriminator(fake_pair)

            loss_d, d_metrics = criterion.get_discriminator_loss(d_real_preds, d_fake_preds)
            loss_d.backward()
            optimizer_d.step()

            # ---------------------------------------------------------------
            # 2. Train Generator: Maximize log(D(x, G(x))) + lambda * L1(y, G(x))
            # ---------------------------------------------------------------
            optimizer_g.zero_grad()

            fake_pair_for_g = torch.cat([l_batch, ab_fake], dim=1)
            d_fake_for_g = discriminator(fake_pair_for_g)

            loss_g, g_metrics = criterion.get_generator_loss(d_fake_for_g, ab_fake, ab_batch)
            loss_g.backward()
            optimizer_g.step()

            running_d_loss += loss_d.item()
            running_g_loss += loss_g.item()
            running_l1_loss += g_metrics["g_loss_l1"]

            if (step + 1) % args.log_interval == 0 or (step + 1) == len(dataloader):
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] | Step [{step+1:04d}/{len(dataloader):04d}] "
                    f"| Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f} "
                    f"| L1: {g_metrics['g_loss_l1']:.4f} | Adv: {g_metrics['g_loss_adv']:.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()

        epoch_time = time.time() - epoch_start
        avg_d = running_d_loss / max(1, len(dataloader))
        avg_g = running_g_loss / max(1, len(dataloader))
        avg_l1 = running_l1_loss / max(1, len(dataloader))
        print(
            f"--> Epoch {epoch} finished in {epoch_time:.1f}s | Avg Loss D: {avg_d:.4f} | Avg Loss G: {avg_g:.4f} | Avg L1: {avg_l1:.4f}"
        )

        # Checkpoint saving
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_file = save_dir / f"chromagan_epoch_{epoch:03d}.pth"
            latest_file = save_dir / "chromagan_latest.pth"
            ckpt_data = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": optimizer_g.state_dict(),
                "opt_d": optimizer_d.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt_data, ckpt_file)
            torch.save(ckpt_data, latest_file)
            print(f"[ChromaGAN Trainer] Saved checkpoint to {ckpt_file}")

            # Save qualitative visual comparison
            generator.eval()
            with torch.no_grad():
                sample_l = l_batch[:1]
                sample_ab_real = ab_batch[:1]
                sample_ab_fake = generator(sample_l)

                l_np = ((sample_l[0, 0].cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
                ab_fake_np = ((sample_ab_fake[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
                ab_real_np = ((sample_ab_real[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)

                lab_fake = cv2.merge([l_np, ab_fake_np[:, :, 0], ab_fake_np[:, :, 1]])
                lab_real = cv2.merge([l_np, ab_real_np[:, :, 0], ab_real_np[:, :, 1]])

                rgb_fake = cv2.cvtColor(lab_fake, cv2.COLOR_LAB2BGR)
                rgb_real = cv2.cvtColor(lab_real, cv2.COLOR_LAB2BGR)
                b_and_w = cv2.cvtColor(l_np, cv2.COLOR_GRAY2BGR)

                comparison = np.hstack([b_and_w, rgb_fake, rgb_real])
                cv2.imwrite(str(sample_dir / f"sample_epoch_{epoch:03d}.png"), comparison)

    print(f"[ChromaGAN Trainer] Training successfully concluded. Models saved in {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChromaGAN cGAN Training Suite")
    parser.add_argument("--data_dir", type=str, default="assets/samples", help="Directory containing training images")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Target image dimension (256x256)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 momentum")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight of L1 chrominance loss")
    parser.add_argument("--decay_step", type=int, default=10, help="Epoch step to decay learning rate")
    parser.add_argument("--save_interval", type=int, default=5, help="Epoch interval to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=5, help="Step interval to print progress")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap max dataset samples")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
