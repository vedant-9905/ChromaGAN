export type WindowPresetKey = "lung" | "soft_tissue" | "bone" | "brain" | "custom";

export interface WindowPreset {
  name: string;
  windowLevel: number; // WL
  windowWidth: number; // WW
  backendKey: "lung" | "soft_tissue" | "default";
}

export interface MetricData {
  psnr: number;
  ssim: number;
  reconstructionFidelity: number;
  inferenceTimeMs: number;
}

export interface SystemHealth {
  status: "ok" | "degraded" | "offline";
  modelLoaded: boolean;
  device: string;
  uptimeSeconds: number;
  gpuName?: string;
  torchVersion?: string;
}

export interface DenoiseResult {
  originalB64: string;
  denoisedB64: string;
  heatmapB64: string;
  metrics: MetricData;
}

export type PresetArtifactType = "normal" | "high_noise" | "streak_artifact";
