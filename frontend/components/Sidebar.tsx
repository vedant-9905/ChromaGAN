"use client";

import React, { useRef } from "react";
import {
  Upload,
  Sliders,
  Sparkles,
  Layers,
  FileImage,
  RefreshCw,
  SunMedium,
  Contrast,
  Zap,
} from "lucide-react";
import { PresetArtifactType, WindowPresetKey, WindowPreset } from "@/types";

interface SidebarProps {
  onFileUpload: (file: File) => void;
  onPresetSelect: (preset: PresetArtifactType) => void;
  selectedPreset: PresetArtifactType | null;
  currentWindowPreset: WindowPresetKey;
  onWindowPresetChange: (key: WindowPresetKey) => void;
  windowLevel: number;
  windowWidth: number;
  onWindowLevelChange: (val: number) => void;
  onWindowWidthChange: (val: number) => void;
  onRunDenoise: () => void;
  isProcessing: boolean;
  hasScanLoaded: boolean;
}

export const WINDOW_PRESET_MAP: Record<WindowPresetKey, WindowPreset> = {
  lung: {
    name: "Lung (Parenchyma)",
    windowLevel: -600,
    windowWidth: 1500,
    backendKey: "lung",
  },
  soft_tissue: {
    name: "Mediastinum / Soft Tissue",
    windowLevel: 40,
    windowWidth: 400,
    backendKey: "soft_tissue",
  },
  bone: {
    name: "Osseous / Bone",
    windowLevel: 400,
    windowWidth: 1800,
    backendKey: "default",
  },
  brain: {
    name: "Cerebral / Brain",
    windowLevel: 40,
    windowWidth: 80,
    backendKey: "default",
  },
  custom: {
    name: "Custom HU Window",
    windowLevel: 0,
    windowWidth: 1000,
    backendKey: "default",
  },
};

export const Sidebar: React.FC<SidebarProps> = ({
  onFileUpload,
  onPresetSelect,
  selectedPreset,
  currentWindowPreset,
  onWindowPresetChange,
  windowLevel,
  windowWidth,
  onWindowLevelChange,
  onWindowWidthChange,
  onRunDenoise,
  isProcessing,
  hasScanLoaded,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <aside className="w-80 border-r border-pacs-800 bg-pacs-900/95 flex flex-col justify-between p-4 overflow-y-auto">
      <div className="space-y-6">
        {/* Upload Panel */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-cyan-400" /> Ingest CT Slice
            </span>
            <span className="text-[10px] text-slate-400 font-mono">DICOM / PNG</span>
          </div>

          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-pacs-700 hover:border-cyan-500/80 rounded-xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 bg-pacs-850/50 hover:bg-pacs-850 group"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/tiff"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-10 h-10 rounded-full bg-pacs-800 group-hover:bg-cyan-500/10 flex items-center justify-center mb-2 transition-colors">
              <FileImage className="w-5 h-5 text-slate-400 group-hover:text-cyan-400 transition-colors" />
            </div>
            <p className="text-xs font-medium text-slate-200">
              Drop slice or <span className="text-cyan-400 underline">browse</span>
            </p>
            <p className="text-[10px] text-slate-400 mt-1">Single-channel 16-bit or 8-bit axial scan</p>
          </div>
        </section>

        {/* Diagnostic Preset Selections */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" /> Anatomical Presets
            </span>
            <span className="text-[10px] text-cyan-400/90 font-mono">1-Click Test</span>
          </div>

          <div className="space-y-2">
            <button
              onClick={() => onPresetSelect("normal")}
              className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex items-center justify-between ${
                selectedPreset === "normal"
                  ? "bg-cyan-950/40 border-cyan-500 text-cyan-200 shadow-glow-sm"
                  : "bg-pacs-850 border-pacs-700/80 text-slate-300 hover:border-pacs-600"
              }`}
            >
              <div>
                <div className="font-semibold text-slate-200">Normal Dose CT (Ref)</div>
                <div className="text-[10px] text-slate-400">High mAs baseline thoracic acquisition</div>
              </div>
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
            </button>

            <button
              onClick={() => onPresetSelect("high_noise")}
              className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex items-center justify-between ${
                selectedPreset === "high_noise"
                  ? "bg-cyan-950/40 border-cyan-500 text-cyan-200 shadow-glow-sm"
                  : "bg-pacs-850 border-pacs-700/80 text-slate-300 hover:border-pacs-600"
              }`}
            >
              <div>
                <div className="font-semibold text-slate-200">Ultra-Low Dose (Poisson Noise)</div>
                <div className="text-[10px] text-slate-400">Quantum mottle & electronic sensor noise</div>
              </div>
              <span className="w-2 h-2 rounded-full bg-amber-400" />
            </button>

            <button
              onClick={() => onPresetSelect("streak_artifact")}
              className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex items-center justify-between ${
                selectedPreset === "streak_artifact"
                  ? "bg-cyan-950/40 border-cyan-500 text-cyan-200 shadow-glow-sm"
                  : "bg-pacs-850 border-pacs-700/80 text-slate-300 hover:border-pacs-600"
              }`}
            >
              <div>
                <div className="font-semibold text-slate-200">Photon Starvation Artifacts</div>
                <div className="text-[10px] text-slate-400">Severe streak lines & beam-hardening</div>
              </div>
              <span className="w-2 h-2 rounded-full bg-rose-400" />
            </button>
          </div>
        </section>

        {/* Hounsfield Window & Leveling */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" /> HU Windowing (W/L)
            </span>
          </div>

          {/* Quick Preset Buttons */}
          <div className="grid grid-cols-2 gap-1.5 mb-3">
            {(["lung", "soft_tissue", "bone", "brain"] as WindowPresetKey[]).map((key) => {
              const preset = WINDOW_PRESET_MAP[key];
              const isSelected = currentWindowPreset === key;
              return (
                <button
                  key={key}
                  onClick={() => onWindowPresetChange(key)}
                  className={`px-2 py-1.5 rounded text-[11px] font-medium border text-center transition-colors truncate ${
                    isSelected
                      ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                      : "bg-pacs-850 border-pacs-700/60 text-slate-300 hover:border-pacs-600"
                  }`}
                >
                  {preset.name.split(" ")[0]}
                </button>
              );
            })}
          </div>

          {/* Sliders */}
          <div className="space-y-3 bg-pacs-850/60 p-3 rounded-lg border border-pacs-800">
            <div>
              <div className="flex justify-between text-[11px] mb-1 font-mono">
                <span className="text-slate-400 flex items-center gap-1">
                  <SunMedium className="w-3 h-3 text-cyan-400" /> Level (WL):
                </span>
                <span className="text-cyan-300 font-semibold">{windowLevel} HU</span>
              </div>
              <input
                type="range"
                min="-1000"
                max="1000"
                step="10"
                value={windowLevel}
                onChange={(e) => onWindowLevelChange(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1 font-mono">
                <span className="text-slate-400 flex items-center gap-1">
                  <Contrast className="w-3 h-3 text-cyan-400" /> Width (WW):
                </span>
                <span className="text-cyan-300 font-semibold">{windowWidth} HU</span>
              </div>
              <input
                type="range"
                min="50"
                max="3000"
                step="25"
                value={windowWidth}
                onChange={(e) => onWindowWidthChange(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </section>
      </div>

      {/* Primary Action Button */}
      <div className="pt-4 border-t border-pacs-800">
        <button
          onClick={onRunDenoise}
          disabled={!hasScanLoaded || isProcessing}
          className={`w-full py-3 px-4 rounded-xl font-semibold text-xs tracking-wider uppercase transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg ${
            !hasScanLoaded || isProcessing
              ? "bg-pacs-800 text-slate-500 cursor-not-allowed border border-pacs-700/50"
              : "bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-pacs-950 font-bold shadow-glow hover:shadow-cyan-500/40"
          }`}
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-slate-300" />
              <span className="text-slate-200">Restoring Anatomy...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Synthesize WGAN-GP</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
