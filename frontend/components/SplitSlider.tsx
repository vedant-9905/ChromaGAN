"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { SlidersHorizontal, Eye, Sparkles } from "lucide-react";

interface SplitSliderProps {
  originalUrl: string;
  denoisedUrl: string;
}

export const SplitSlider: React.FC<SplitSliderProps> = ({
  originalUrl,
  denoisedUrl,
}) => {
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage [0, 100]
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
    <div className="w-full flex-1 flex flex-col p-4 bg-pacs-950 overflow-hidden">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-pacs-800/80">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" /> Split-Screen Micro-Structure Inspector
          </span>
          <span className="text-[11px] font-mono text-cyan-400 bg-pacs-850 px-2 py-0.5 rounded border border-pacs-700">
            Drag divider to inspect edge retention & artifact suppression
          </span>
        </div>
        <div className="text-xs font-mono text-slate-400">
          Split Position: <span className="text-cyan-300 font-bold">{Math.round(sliderPos)}%</span>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center min-h-0">
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          className="relative max-h-[65vh] w-auto aspect-square bg-black rounded-xl border border-pacs-800 overflow-hidden shadow-card cursor-ew-resize select-none"
        >
          {/* Base Image: Denoised (Right Side / Full) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={denoisedUrl}
            alt="Denoised Output"
            className="w-full h-full object-contain pointer-events-none"
          />

          {/* Overlay Image: Low-Dose Original (Left Side clipped by slider position) */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ width: `${sliderPos}%` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={originalUrl}
              alt="Low-Dose Input"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none max-w-none"
              style={{
                width: containerRef.current?.clientWidth ?? "100%",
                height: containerRef.current?.clientHeight ?? "100%",
              }}
            />
          </div>

          {/* Draggable Vertical Divider Bar */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-glow pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            {/* Centered Grab Handle */}
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-cyan-500 text-pacs-950 flex items-center justify-center shadow-lg border-2 border-white pointer-events-auto cursor-ew-resize">
              <SlidersHorizontal className="w-3.5 h-3.5" />
            </div>
          </div>

          {/* Left Label: Low-Dose Input */}
          <div className="absolute top-3 left-3 bg-pacs-900/80 backdrop-blur-sm border border-amber-500/40 px-2 py-1 rounded text-[11px] font-mono text-amber-300 flex items-center gap-1.5 pointer-events-none">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            <span>Low-Dose CT (Noisy)</span>
          </div>

          {/* Right Label: Denoised WGAN */}
          <div className="absolute top-3 right-3 bg-pacs-900/80 backdrop-blur-sm border border-cyan-500/40 px-2 py-1 rounded text-[11px] font-mono text-cyan-300 flex items-center gap-1.5 pointer-events-none">
            <Sparkles className="w-3 h-3 text-cyan-400" />
            <span>RadSwin-WGAN Reconstructed</span>
          </div>
        </div>
      </div>
    </div>
  );
};
