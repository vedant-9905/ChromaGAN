"use client";

import React, { useState, useRef, useCallback } from "react";
import { SlidersHorizontal, Sparkles, Image as ImageIcon, ZoomIn, ZoomOut } from "lucide-react";

interface SplitSliderProps {
  originalUrl: string;
  colorizedUrl: string;
}

export const SplitSlider: React.FC<SplitSliderProps> = ({
  originalUrl,
  colorizedUrl,
}) => {
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage [0, 100]
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef<boolean>(false);

  const handlePointerDown = (e: React.PointerEvent) => {
    isDraggingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    updateSlider(e.clientX);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDraggingRef.current) return;
    updateSlider(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    isDraggingRef.current = false;
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  const updateSlider = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  }, []);

  return (
    <div className="w-full flex-1 flex flex-col p-4 bg-slate-950 overflow-hidden">
      {/* Viewport Top Bar */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-violet-400" /> Interactive Split Comparison
          </span>
          <span className="text-[11px] font-mono text-violet-300 bg-slate-850 px-2 py-0.5 rounded-full border border-slate-700">
            Drag divider to inspect chromatic realism
          </span>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center bg-slate-850 rounded-lg border border-slate-700 px-1 py-0.5 space-x-1">
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.75, z - 0.25))}
              className="p-1 text-slate-400 hover:text-slate-200 rounded"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono text-slate-300 px-1">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.25))}
              className="p-1 text-slate-400 hover:text-slate-200 rounded"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoomLevel(1)}
              className="text-[10px] text-violet-400 hover:underline px-1 border-l border-slate-700 font-mono"
            >
              1:1
            </button>
          </div>
          <div className="text-xs font-mono text-slate-400">
            Split: <span className="text-violet-300 font-bold">{Math.round(sliderPos)}%</span>
          </div>
        </div>
      </div>

      {/* Main Comparative Canvas Viewport */}
      <div className="flex-1 flex items-center justify-center min-h-0 overflow-hidden">
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={{ transform: `scale(${zoomLevel})` }}
          className="relative max-h-[68vh] w-auto aspect-square bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl cursor-ew-resize select-none transition-transform duration-150"
        >
          {/* Base Layer: Colorized Full Result */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={colorizedUrl}
            alt="Colorized Output"
            className="w-full h-full object-contain pointer-events-none"
          />

          {/* Overlay Layer: Grayscale B&W Original (Clipped by slider width) */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ width: `${sliderPos}%` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={originalUrl}
              alt="Grayscale Original"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none max-w-none"
              style={{
                width: containerRef.current?.clientWidth ?? "100%",
                height: containerRef.current?.clientHeight ?? "100%",
              }}
            />
          </div>

          {/* Draggable Divider Bar */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-violet-400 shadow-lg shadow-violet-500/50 pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-violet-600 text-white flex items-center justify-center shadow-xl border-2 border-white pointer-events-auto cursor-ew-resize">
              <SlidersHorizontal className="w-4 h-4" />
            </div>
          </div>

          {/* Left Tag: Original B&W */}
          <div className="absolute top-3.5 left-3.5 bg-slate-900/80 backdrop-blur-md border border-slate-700/80 px-2.5 py-1 rounded-lg text-xs font-mono text-slate-300 flex items-center gap-1.5 pointer-events-none shadow-md">
            <ImageIcon className="w-3.5 h-3.5 text-slate-400" />
            <span>Black & White Original</span>
          </div>

          {/* Right Tag: ChromaGAN Colorized */}
          <div className="absolute top-3.5 right-3.5 bg-violet-950/80 backdrop-blur-md border border-violet-500/40 px-2.5 py-1 rounded-lg text-xs font-mono text-violet-200 flex items-center gap-1.5 pointer-events-none shadow-md">
            <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            <span>ChromaGAN cGAN Restored</span>
          </div>
        </div>
      </div>
    </div>
  );
};
