# RadSwin-WGAN Radiologist Workstation (Frontend)

Next.js 14 medical imaging workstation for deep learning LDCT restoration.

## Features
- **3-Way Comparative Viewport**: Low-Dose CT (Noisy), WGAN-GP Reconstructed (Denoised), and Jet-Colormap Residual Error Heatmap.
- **Micro-Structure Split-Screen Inspector**: Draggable comparative divider for examining anatomical fine edge retention vs. noise rejection.
- **Hounsfield Unit (HU) Windowing**: Real-time Window Level (WL) and Window Width (WW) sliders with presets for Lung, Soft Tissue, Bone, and Brain.
- **Anatomical Presets**: One-click procedural synthetic CT generation (Normal dose, Poisson quantum noise, Photon starvation streak artifacts).
- **Quantitative Metrics HUD**: Real-time display of Peak SNR (PSNR), Structural Similarity (SSIM), Reconstruction Fidelity %, and GPU/CPU inference latency.

## Getting Started

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.
