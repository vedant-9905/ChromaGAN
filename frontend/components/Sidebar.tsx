"use client";

import React, { useRef } from "react";
import {
  Upload,
  Sparkles,
  Sliders,
  Image as ImageIcon,
  Sun,
  Layers,
  Wand2,
  RefreshCw,
} from "lucide-react";
import { ColorStyle, ColorStyleOption, SamplePresetKey } from "@/types";

interface SidebarProps {
  onFileUpload: (file: File) => void;
  onPresetSelect: (preset: SamplePresetKey) => void;
  selectedPreset: SamplePresetKey | null;
  selectedStyle: ColorStyle;
  onStyleSelect: (style: ColorStyle) => void;
  saturation: number;
  onSaturationChange: (val: number) => void;
  onRunColorize: () => void;
  isProcessing: boolean;
  hasPhotoLoaded: boolean;
}

const COLOR_STYLES: ColorStyleOption[] = [
  {
    id: "vibrant",
    label: "Vibrant Modern",
    description: "Rich chromatic saturation with high dynamic range",
    tagColor: "border-purple-500/40 text-purple-300",
  },
  {
    id: "natural",
    label: "Natural Historic",
    description: "Balanced, historically authentic archival tones",
    tagColor: "border-emerald-500/40 text-emerald-300",
  },
  {
    id: "warm_vintage",
    label: "Warm Golden Hour",
    description: "Sepia-infused golden highlights and warm amber glow",
    tagColor: "border-amber-500/40 text-amber-300",
  },
  {
    id: "cool_cinematic",
    label: "Cool Cinematic",
    description: "Teal and slate tones with dramatic cool shadows",
    tagColor: "border-cyan-500/40 text-cyan-300",
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  onFileUpload,
  onPresetSelect,
  selectedPreset,
  selectedStyle,
  onStyleSelect,
  saturation,
  onSaturationChange,
  onRunColorize,
  isProcessing,
  hasPhotoLoaded,
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
    <aside className="w-80 border-r border-slate-800 bg-slate-900/95 flex flex-col justify-between p-4 overflow-y-auto">
      <div className="space-y-6">
        {/* Upload Zone */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-violet-400" /> Upload B&W Photo
            </span>
            <span className="text-[10px] text-slate-400 font-mono">JPG / PNG / WebP</span>
          </div>

          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-violet-500/80 rounded-2xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 bg-slate-850/50 hover:bg-slate-800/80 group"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="w-10 h-10 rounded-full bg-slate-800 group-hover:bg-violet-500/10 flex items-center justify-center mb-2 transition-colors">
              <ImageIcon className="w-5 h-5 text-slate-400 group-hover:text-violet-400 transition-colors" />
            </div>
            <p className="text-xs font-medium text-slate-200">
              Drag photo here or <span className="text-violet-400 underline">browse</span>
            </p>
            <p className="text-[10px] text-slate-400 mt-1">Vintage archival or grayscale image</p>
          </div>
        </section>

        {/* 1-Click Historical Presets */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-violet-400" /> Historical Presets
            </span>
            <span className="text-[10px] text-violet-400 font-mono">1-Click Test</span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => onPresetSelect("classic_portrait")}
              className={`p-2.5 rounded-xl border text-left transition-all ${
                selectedPreset === "classic_portrait"
                  ? "bg-violet-950/40 border-violet-500 text-violet-200 shadow-md shadow-violet-500/10"
                  : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-600"
              }`}
            >
              <div className="text-xs font-semibold">1920s Portrait</div>
              <div className="text-[10px] text-slate-400 truncate">Studio lighting</div>
            </button>

            <button
              onClick={() => onPresetSelect("vintage_car")}
              className={`p-2.5 rounded-xl border text-left transition-all ${
                selectedPreset === "vintage_car"
                  ? "bg-violet-950/40 border-violet-500 text-violet-200 shadow-md shadow-violet-500/10"
                  : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-600"
              }`}
            >
              <div className="text-xs font-semibold">Classic Car</div>
              <div className="text-[10px] text-slate-400 truncate">1930s Roadster</div>
            </button>

            <button
              onClick={() => onPresetSelect("old_city")}
              className={`p-2.5 rounded-xl border text-left transition-all ${
                selectedPreset === "old_city"
                  ? "bg-violet-950/40 border-violet-500 text-violet-200 shadow-md shadow-violet-500/10"
                  : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-600"
              }`}
            >
              <div className="text-xs font-semibold">Historic City</div>
              <div className="text-[10px] text-slate-400 truncate">Cobblestone street</div>
            </button>

            <button
              onClick={() => onPresetSelect("landscape")}
              className={`p-2.5 rounded-xl border text-left transition-all ${
                selectedPreset === "landscape"
                  ? "bg-violet-950/40 border-violet-500 text-violet-200 shadow-md shadow-violet-500/10"
                  : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-600"
              }`}
            >
              <div className="text-xs font-semibold">Yosemite Lake</div>
              <div className="text-[10px] text-slate-400 truncate">Alpine peaks</div>
            </button>
          </div>
        </section>

        {/* Color Grading Styles */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-violet-400" /> Color Palette Grade
            </span>
          </div>

          <div className="space-y-1.5">
            {COLOR_STYLES.map((style) => (
              <button
                key={style.id}
                onClick={() => onStyleSelect(style.id)}
                className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all flex flex-col ${
                  selectedStyle === style.id
                    ? "bg-violet-950/40 border-violet-500 text-violet-200 shadow-sm"
                    : "bg-slate-800/50 border-slate-700/70 text-slate-300 hover:border-slate-600"
                }`}
              >
                <span className="font-semibold">{style.label}</span>
                <span className="text-[10px] text-slate-400 mt-0.5">{style.description}</span>
              </button>
            ))}
          </div>
        </section>

        {/* Saturation Slider */}
        <section>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sun className="w-3.5 h-3.5 text-violet-400" /> Chromatic Saturation
            </span>
            <span className="text-xs font-mono text-violet-300 font-bold">{saturation.toFixed(1)}x</span>
          </div>

          <div className="bg-slate-800/60 p-3 rounded-xl border border-slate-700/80">
            <input
              type="range"
              min="0.5"
              max="2.2"
              step="0.1"
              value={saturation}
              onChange={(e) => onSaturationChange(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-[9px] font-mono text-slate-400 mt-1.5">
              <span>Muted (0.5x)</span>
              <span>Default (1.2x)</span>
              <span>Intense (2.2x)</span>
            </div>
          </div>
        </section>
      </div>

      {/* Main Colorize Action Button */}
      <div className="pt-4 border-t border-slate-800">
        <button
          onClick={onRunColorize}
          disabled={!hasPhotoLoaded || isProcessing}
          className={`w-full py-3.5 px-4 rounded-xl font-bold text-xs tracking-wider uppercase transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg ${
            !hasPhotoLoaded || isProcessing
              ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50"
              : "bg-gradient-to-r from-violet-600 via-purple-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white shadow-purple-500/30 hover:shadow-purple-500/50"
          }`}
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
              <span>Synthesizing Colors...</span>
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              <span>Colorize with cGAN</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
