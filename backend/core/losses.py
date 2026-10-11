"""
core/losses.py
==============
Loss suite for ChromaGAN (Conditional GAN / Pix2Pix):
1. Adversarial Loss (BCEWithLogitsLoss)
2. L1 Chrominance Loss (|ab_true - ab_pred|_1 with lambda=100)
3. Combined Generator and Discriminator Loss functions
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class ChromaGANLoss(nn.Module):
    """
    Combined Conditional GAN Loss:
    L_total = L_cGAN(G, D) + lambda_L1 * L1(ab_real, ab_fake)
    """

    def __init__(self, lambda_l1: float = 100.0) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()

    def get_discriminator_loss(
        self,
        d_real_preds: torch.Tensor,
        d_fake_preds: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates PatchGAN Discriminator loss:
        L_D = 0.5 * (BCE(D(real), 1) + BCE(D(fake), 0))
        """
        real_targets = torch.ones_like(d_real_preds)
        fake_targets = torch.zeros_like(d_fake_preds)

        loss_real = self.adversarial_criterion(d_real_preds, real_targets)
        loss_fake = self.adversarial_criterion(d_fake_preds, fake_targets)
        total_d_loss = 0.5 * (loss_real + loss_fake)

        return total_d_loss, {
            "d_loss_real": loss_real.item(),
            "d_loss_fake": loss_fake.item(),
            "d_loss_total": total_d_loss.item(),
        }

    def get_generator_loss(
        self,
        d_fake_preds: torch.Tensor,
        ab_fake: torch.Tensor,
        ab_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates U-Net Generator loss:
        L_G = BCE(D(G(L)), 1) + lambda_L1 * L1(ab_real, ab_fake)
        """
        real_targets = torch.ones_like(d_fake_preds)
        adv_loss = self.adversarial_criterion(d_fake_preds, real_targets)
        l1_loss = self.l1_criterion(ab_fake, ab_real)

        total_g_loss = adv_loss + self.lambda_l1 * l1_loss

        return total_g_loss, {
            "g_loss_adv": adv_loss.item(),
            "g_loss_l1": l1_loss.item(),
            "g_loss_total": total_g_loss.item(),
        }
