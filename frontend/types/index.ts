export type ColorStyle = "vibrant" | "natural" | "warm_vintage" | "cool_cinematic";

export type SamplePresetKey = "vintage_car" | "classic_portrait" | "old_city" | "landscape";

export interface ColorStyleOption {
  id: ColorStyle;
  label: string;
  description: string;
  tagColor: string;
}

export interface MetricData {
  colorfulnessScore: number;
  resolution: string;
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

export interface ColorizeResult {
  originalB64: string;
  colorizedB64: string;
  heatmapB64: string;
  metrics: MetricData;
}
