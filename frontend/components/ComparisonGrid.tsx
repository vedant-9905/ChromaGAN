"use client";

import React, { useState } from "react";
import { ZoomIn, ZoomOut, Maximize2, Crosshair, HelpCircle, Activity } from "lucide-react";

interface ComparisonGridProps {
  originalUrl: string | null;
  denoisedUrl: string | null;
  heatmapUrl: string | null;
  windowInfo: string;
  isProcessing: boolean;
}

export const ComparisonGrid: React.FC<ComparisonGridProps> = ({
  originalUrl,
  denoisedUrl,
  heatmapUrl,
  windowInfo,
  isProcessing,
}) => {
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [showCrosshair, setShowCrosshair] = useState<boolean>(true);

  const handleZoomIn = () => setZoomLevel((prev) => Math.min(prev + 0.25, 2.5));
  const handleZoomOut = () => setZoomLevel((prev) => Math.max(prev - 0.25, 0.75));
  const handleResetZoom = () => setZoomLevel(1);

  return (
    <div className="flex-1 flex flex-col h-full bg-pacs-950 p-4 overflow-hidden">
      {/* Top Toolbar */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-pacs-800/80">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            3-Way Tri-Planar Viewport
          </span>
          <span className="text-[11px] font-mono text-cyan-400 bg-pacs-850 px-2 py-0.5 rounded border border-pacs-700">
            {windowInfo}
          </span>
        </div>

        {/* View Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowCrosshair(!showCrosshair)}
            title="Toggle Anatomical Crosshair"
            className={`p-1.5 rounded border text-xs transition-colors ${
              showCrosshair
                ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                : "bg-pacs-850 border-pacs-700 text-slate-400 hover:text-slate-200"
            }`}
          >
            <Crosshair className="w-3.5 h-3.5" />
          </button>
          <div className="flex items-center bg-pacs-850 rounded border border-pacs-700 px-1 py-0.5 space-x-1">
            <button
              onClick={handleZoomOut}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-pacs-750 rounded"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono text-slate-300 px-1">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-1 text-slate-400 hover:text-slate-200 hover:bg-pacs-750 rounded"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleResetZoom}
              className="text-[10px] text-cyan-400 hover:underline px-1 border-l border-pacs-700 font-mono"
            >
              1:1
            </button>
          </div>
        </div>
      </div>

      {/* 3-Way Grid Panes */}
      <div className="grid grid-cols-3 gap-4 flex-1 h-[calc(100%-48px)] min-h-0">
        {/* Pane 1: Low-Dose Input */}
        <div className="flex flex-col bg-pacs-900 rounded-xl border border-pacs-800 overflow-hidden shadow-card relative group">
          {/* Pane Header */}
          <div className="px-3 py-2 bg-pacs-850 border-b border-pacs-800 flex items-center justify-between z-10">
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span className="text-xs font-bold text-slate-200 tracking-wide">
                1. Low-Dose CT (Input)
              </span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">Degraded / Noisy</span>
          </div>

          {/* Canvas / View Area */}
          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden relative cursor-crosshair">
            {originalUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out relative"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={originalUrl}
                  alt="Low-Dose Input"
                  className="max-h-[60vh] max-w-full object-contain pointer-events-none select-none"
                />
                {showCrosshair && (
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute inset-x-0 top-1/2 h-[1px] bg-cyan-500/30" />
                    <div className="absolute inset-y-0 left-1/2 w-[1px] bg-cyan-500/30" />
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-30 animate-pulse" />
                <p className="text-xs font-medium">Select a Preset or Upload CT Slice</p>
              </div>
            )}

            {/* In-viewport DICOM Overlay HUD */}
            <div className="absolute top-2 left-2 pointer-events-none font-mono text-[9px] text-slate-400/80 space-y-0.5">
              <p>AXIAL THORACIC</p>
              <p>KV: 80 | MAS: 25</p>
            </div>
            <div className="absolute bottom-2 right-2 pointer-events-none font-mono text-[9px] text-slate-400/80 text-right">
              <p>FOV: 350mm</p>
              <p>THICK: 1.0mm</p>
            </div>
          </div>
        </div>

        {/* Pane 2: WGAN-GP Reconstructed Output */}
        <div className="flex flex-col bg-pacs-900 rounded-xl border border-cyan-500/40 overflow-hidden shadow-glow-sm relative group">
          {/* Pane Header */}
          <div className="px-3 py-2 bg-gradient-to-r from-cyan-950/60 to-pacs-850 border-b border-cyan-500/30 flex items-center justify-between z-10">
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-xs font-bold text-cyan-200 tracking-wide">
                2. RadSwin-WGAN (Denoised)
              </span>
            </div>
            <span className="text-[10px] font-mono text-cyan-400">Restored Anatomy</span>
          </div>

          {/* Canvas / View Area */}
          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden relative cursor-crosshair">
            {denoisedUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out relative"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={denoisedUrl}
                  alt="WGAN-GP Reconstructed"
                  className="max-h-[60vh] max-w-full object-contain pointer-events-none select-none"
                />
                {showCrosshair && (
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute inset-x-0 top-1/2 h-[1px] bg-cyan-400/40" />
                    <div className="absolute inset-y-0 left-1/2 w-[1px] bg-cyan-400/40" />
                  </div>
                )}
              </div>
            ) : isProcessing ? (
              <div className="text-center p-6 text-cyan-400">
                <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                <p className="text-xs font-medium animate-pulse">Running ResUNet Generator...</p>
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <p className="text-xs font-medium">Awaiting Inference Execution</p>
              </div>
            )}

            {/* In-viewport DICOM Overlay HUD */}
            <div className="absolute top-2 left-2 pointer-events-none font-mono text-[9px] text-cyan-400/70 space-y-0.5">
              <p>RESUNET + WGAN-GP</p>
              <p>MS-SSIM + VGG-19 LOSS</p>
            </div>
            <div className="absolute bottom-2 right-2 pointer-events-none font-mono text-[9px] text-emerald-400/80 text-right">
              <p>NOISE REJECTED</p>
              <p>EDGES PRESERVED</p>
            </div>
          </div>
        </div>

        {/* Pane 3: Residual / Difference Heatmap */}
        <div className="flex flex-col bg-pacs-900 rounded-xl border border-pacs-800 overflow-hidden shadow-card relative group">
          {/* Pane Header */}
          <div className="px-3 py-2 bg-pacs-850 border-b border-pacs-800 flex items-center justify-between z-10">
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span className="text-xs font-bold text-slate-200 tracking-wide">
                3. Residual Artifact Heatmap
              </span>
            </div>
            <span className="text-[10px] font-mono text-rose-300">|Input - Denoised|</span>
          </div>

          {/* Canvas / View Area */}
          <div className="flex-1 bg-black flex items-center justify-center overflow-hidden relative cursor-crosshair">
            {heatmapUrl ? (
              <div
                style={{ transform: `scale(${zoomLevel})` }}
                className="transition-transform duration-150 ease-out relative"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={heatmapUrl}
                  alt="Residual Heatmap"
                  className="max-h-[60vh] max-w-full object-contain pointer-events-none select-none"
                />
                {showCrosshair && (
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute inset-x-0 top-1/2 h-[1px] bg-rose-500/30" />
                    <div className="absolute inset-y-0 left-1/2 w-[1px] bg-rose-500/30" />
                  </div>
                )}
              </div>
            ) : isProcessing ? (
              <div className="text-center p-6 text-slate-500">
                <p className="text-xs font-medium animate-pulse">Calculating difference map...</p>
              </div>
            ) : (
              <div className="text-center p-6 text-slate-500">
                <p className="text-xs font-medium">Residual Heatmap Generated on Run</p>
              </div>
            )}

            {/* Jet Colormap Scale Legend */}
            {heatmapUrl && (
              <div className="absolute bottom-2 left-2 pointer-events-none bg-pacs-900/90 border border-pacs-700/80 rounded p-1.5 shadow-sm">
                <div className="text-[8px] font-mono text-slate-300 mb-1 flex justify-between">
                  <span>0 (Low)</span>
                  <span className="text-rose-400">Max Noise</span>
                </div>
                <div className="w-28 h-2 rounded bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-400 to-red-600 border border-pacs-600" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
