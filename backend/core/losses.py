"""
core/losses.py
==============
Complete loss suite for RadSwin-WGAN training:

  1. WGAN-GP gradient penalty (λ=10)
  2. L1 pixel loss
  3. MS-SSIM structural loss (differentiable)
  4. Perceptual loss via VGG-19 relu5_2 features

Combined objective:
  L_total = L_WGAN + α·L1 + β·L_SSIM + γ·L_perceptual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. WGAN Gradient Penalty
# ---------------------------------------------------------------------------

def compute_gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Computes the 1-Lipschitz gradient penalty for WGAN-GP.

    Interpolates uniformly between real and fake samples, passes through
    the discriminator, and penalizes the L2 norm of gradients deviating
    from 1.

    Args:
        discriminator : the critic network
        real          : batch of real images  [B, C, H, W]
        fake          : batch of generated images [B, C, H, W]
        device        : computation device
        lambda_gp     : penalty coefficient (default 10.0)

    Returns:
        Scalar gradient penalty tensor
    """
    batch_size = real.size(0)

    # Uniform interpolation coefficient α ∈ [0, 1]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real)

    # Interpolated samples x̂ = α·real + (1-α)·fake
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    # Critic scores for interpolated samples
    d_interpolated = discriminator(interpolated)

    # Gradient of critic output w.r.t. interpolated input
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten spatial dims; compute per-sample gradient L2 norms
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)

    # Penalty = E[(||∇||₂ - 1)²]
    penalty = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return penalty


def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Generator tries to maximize critic score → minimize -E[D(G(z))]"""
    return -fake_scores.mean()


def wasserstein_discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """Critic maximizes E[D(real)] - E[D(fake)] → minimize reverse"""
    return fake_scores.mean() - real_scores.mean()


# ---------------------------------------------------------------------------
# 2. MS-SSIM Loss (differentiable PyTorch implementation)
# ---------------------------------------------------------------------------

def _gaussian_kernel(
    kernel_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
) -> torch.Tensor:
    """Creates a 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # 1 x 1 x K x K
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel_2d.contiguous()


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Returns the per-patch SSIM map for tensors x and y."""
    channels = x.size(1)
    mu_x = F.conv2d(x, kernel, groups=channels, padding=kernel.size(-1) // 2)
    mu_y = F.conv2d(y, kernel, groups=channels, padding=kernel.size(-1) // 2)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, groups=channels, padding=kernel.size(-1) // 2) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, groups=channels, padding=kernel.size(-1) // 2) - mu_y_sq
    sigma_xy   = F.conv2d(x * y, kernel, groups=channels, padding=kernel.size(-1) // 2) - mu_xy

    # Ensure non-negative variances
    sigma_x_sq = sigma_x_sq.clamp(min=0)
    sigma_y_sq = sigma_y_sq.clamp(min=0)

    numerator   = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    return numerator / (denominator + 1e-8)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss: L_SSIM = 1 - SSIM(x, y)

    Operates on single-channel images in [-1, 1] range.
    Images are internally rescaled to [0, 1].
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        kernel = _gaussian_kernel(self.kernel_size, self.sigma, channels).to(pred.device)

        # Rescale from [-1, 1] to [0, 1]
        pred_01   = (pred   + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5

        ssim_map = _ssim_map(pred_01, target_01, kernel)
        return 1.0 - ssim_map.mean()


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss.

    Computes SSIM at 5 scales (as per Wang et al. 2003) and combines
    with the luminance term at the finest scale.
    """

    _WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        kernel = _gaussian_kernel(self.kernel_size, self.sigma, channels).to(pred.device)

        pred_01   = (pred   + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5

        weights = torch.tensor(self._WEIGHTS, device=pred.device, dtype=pred.dtype)
        msssim_val = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        img1, img2 = pred_01, target_01

        for scale_idx in range(len(self._WEIGHTS)):
            ssim_map = _ssim_map(img1, img2, kernel)
            cs_map = ssim_map  # simplified: using full SSIM map at each scale
            msssim_val = msssim_val + weights[scale_idx] * cs_map.mean()

            if scale_idx < len(self._WEIGHTS) - 1:
                img1 = F.avg_pool2d(img1, 2, 2)
                img2 = F.avg_pool2d(img2, 2, 2)

        return 1.0 - msssim_val


# ---------------------------------------------------------------------------
# 3. Perceptual Loss (VGG-19 relu5_2)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-19 features extracted at relu5_2.

    Input single-channel images are replicated to 3 channels for VGG
    compatibility. Features are L1-compared in VGG feature space.

    The VGG-19 weights are frozen; gradients do not flow through VGG.
    """

    # VGG-19 layer indices up to relu5_2 (index 35 in features Sequential)
    _FEATURE_LAYER: int = 35  # relu5_2

    def __init__(self, resize_to_224: bool = False) -> None:
        super().__init__()
        self.resize = resize_to_224

        vgg19 = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        # Freeze VGG completely
        for param in vgg19.parameters():
            param.requires_grad = False

        # Keep only up to relu5_2
        self.features = nn.Sequential(
            *list(vgg19.features.children())[: self._FEATURE_LAYER + 1]
        )
        self.features.eval()

        # ImageNet normalization constants
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale [-1,1] grayscale → [0,1] RGB and ImageNet-normalize."""
        x_01 = (x + 1.0) * 0.5                  # [-1,1] → [0,1]
        x_rgb = x_01.expand(-1, 3, -1, -1)      # 1-ch → 3-ch
        if self.resize:
            x_rgb = F.interpolate(x_rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return (x_rgb - self.mean) / self.std    # ImageNet norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat   = self.features(self._preprocess(pred))
        target_feat = self.features(self._preprocess(target))
        return F.l1_loss(pred_feat, target_feat.detach())


# ---------------------------------------------------------------------------
# 4. Combined Generator Loss
# ---------------------------------------------------------------------------

class GeneratorLoss(nn.Module):
    """
    Combined generator objective:
      L_total = L_WGAN + α·L1 + β·L_SSIM + γ·L_perceptual

    Args:
        alpha   : weight for L1 pixel loss  (default 100.0)
        beta    : weight for SSIM loss      (default 10.0)
        gamma   : weight for perceptual     (default 10.0)
    """

    def __init__(
        self,
        alpha: float = 100.0,
        beta: float  = 10.0,
        gamma: float = 10.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.l1      = nn.L1Loss()
        self.ssim    = MSSSIMLoss()
        self.percept = VGGPerceptualLoss()

    def forward(
        self,
        fake_scores: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            fake_scores : discriminator scores on generated images [B, 1, H', W']
            pred        : generated images [B, 1, H, W]
            target      : ground-truth images [B, 1, H, W]

        Returns:
            total loss (scalar), component dict for logging
        """
        l_wgan    = wasserstein_generator_loss(fake_scores)
        l1        = self.l1(pred, target)
        l_ssim    = self.ssim(pred, target)
        l_percept = self.percept(pred, target)

        total = l_wgan + self.alpha * l1 + self.beta * l_ssim + self.gamma * l_percept

        components = {
            "loss_wgan":       l_wgan.item(),
            "loss_l1":         l1.item(),
            "loss_ssim":       l_ssim.item(),
            "loss_perceptual": l_percept.item(),
            "loss_total":      total.item(),
        }
        return total, components
