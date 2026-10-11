"use client";

import React, { useState } from "react";
import { ZoomIn, ZoomOut, Image as ImageIcon, Sparkles, Activity } from "lucide-react";

interface SideBySideGridProps {
  originalUrl: string | null;
  colorizedUrl: string | null;
  heatmapUrl: string | null;
  isProcessing: boolean;
}

export const SideBySideGrid: React.FC<SideBySideGridProps> = ({
  originalUrl,
  colorizedUrl,
  heatmapUrl,
  isProcessing,
}) => {
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 p-4 overflow-hidden">
      {/* Top Bar */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Tri-View Photographic & Chrominance Studio
          </span>
          <span className="text-[11px] font-mono text-violet-400 bg-slate-850 px-2 py-0.5 rounded-full border border-slate-700">
            Luminance L ➔ Predicted (a, b) Chrominance
          </span>
        </div>

        <div className="flex items-center bg-slate-850 rounded-lg border border-slate-700 px-1 py-0.5 space-x-1">
          <button
            onClick={() => setZoomLevel((z) => Math.max(0.75, z - 0.25))}
            className="p-1 text-slate-400 hover:text-slate-200 rounded"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-[10px] font-mono text-slate-300 px-1">
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.25))}
            className="p-1 text-slate-400 hover:text-slate-200 rounded"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 3 Panels Grid */}
      <div className="grid grid-cols-3 gap-4 flex-1 h-[calc(100%-48px)] min-h-0">
        {/* Panel 1: Original B&W */}
        <div className="flex flex-col bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-lg">
          <div className="px-3.5 py-2.5 bg-slate-850 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-slate-400" />
              <span className="text-xs font-bold text-slate-200">1. Grayscale / B&W Input</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Luminance (L)</span>
          </div>

          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden p-2">
            {originalUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={originalUrl}
                  alt="Original B&W"
                  className="max-h-[58vh] max-w-full object-contain rounded-lg"
                />
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <ImageIcon className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Select a preset or upload an image</p>
              </div>
            )}
          </div>
        </div>

        {/* Panel 2: ChromaGAN Colorized */}
        <div className="flex flex-col bg-slate-900 rounded-2xl border border-violet-500/40 overflow-hidden shadow-lg shadow-violet-500/10">
          <div className="px-3.5 py-2.5 bg-gradient-to-r from-violet-950/70 to-slate-850 border-b border-violet-500/30 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
              <span className="text-xs font-bold text-violet-200">2. ChromaGAN Colorized</span>
            </div>
            <span className="text-[10px] font-mono text-violet-300">cGAN Synthesis</span>
          </div>

          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden p-2">
            {colorizedUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={colorizedUrl}
                  alt="Colorized Result"
                  className="max-h-[58vh] max-w-full object-contain rounded-lg"
                />
              </div>
            ) : isProcessing ? (
              <div className="text-center p-6 text-violet-400">
                <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                <p className="text-xs font-medium animate-pulse">Running U-Net Generator...</p>
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Awaiting cGAN Execution</p>
              </div>
            )}
          </div>
        </div>

        {/* Panel 3: Chromatic Distribution Heatmap */}
        <div className="flex flex-col bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-lg">
          <div className="px-3.5 py-2.5 bg-slate-850 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="text-xs font-bold text-slate-200">3. Chromatic Distribution Map</span>
            </div>
            <span className="text-[10px] font-mono text-cyan-300">Turbo Color Energy</span>
          </div>

          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden p-2">
            {heatmapUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={heatmapUrl}
                  alt="Chromatic Heatmap"
                  className="max-h-[58vh] max-w-full object-contain rounded-lg"
                />
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">Chromatic map generated on run</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
