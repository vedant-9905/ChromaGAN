"use client";

import React from "react";
import { Palette, Cpu, Download, Sparkles, Wand2, Activity } from "lucide-react";
import { MetricData, SystemHealth } from "@/types";

interface HeaderProps {
  systemHealth: SystemHealth;
  metrics: MetricData | null;
  isProcessing: boolean;
  onDownload: () => void;
  hasColorizedResult: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  systemHealth,
  metrics,
  isProcessing,
  onDownload,
  hasColorizedResult,
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-6 flex items-center justify-between z-30 sticky top-0 shadow-sm">
      {/* Brand & System Status */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-600 via-purple-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-purple-500/20">
            <Palette className="w-5 h-5 text-white stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-base tracking-tight text-white">
                Chroma<span className="text-violet-400">GAN</span>
              </span>
              <span className="text-[10px] font-mono uppercase bg-violet-950 text-violet-300 px-2 py-0.5 rounded-full border border-violet-500/30">
                CIELAB cGAN Studio
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">
              Neural Black & White Photo Colorizer & Vintage Restorer
            </p>
          </div>
        </div>

        {/* Vertical Divider */}
        <div className="h-7 w-[1px] bg-slate-800 mx-1" />

        {/* Runtime Status */}
        <div className="flex items-center space-x-2">
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              systemHealth.status === "ok"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : systemHealth.status === "degraded"
                ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                : "bg-purple-500/10 text-purple-300 border-purple-500/30"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                systemHealth.status === "ok"
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-purple-400 animate-pulse"
              }`}
            />
            <span className="capitalize">
              {systemHealth.status === "ok" ? "PyTorch cGAN Online" : "Studio Engine Ready"}
            </span>
          </div>

          <div className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-slate-800/80 border border-slate-700/80 text-slate-300 text-xs font-mono">
            <Cpu className="w-3.5 h-3.5 text-violet-400" />
            <span className="uppercase text-[11px] font-semibold">
              {systemHealth.gpuName ? systemHealth.gpuName : systemHealth.device}
            </span>
          </div>
        </div>
      </div>

      {/* Metrics HUD & Export */}
      <div className="flex items-center space-x-3">
        {/* Colorfulness Metric */}
        <div className="bg-slate-800/80 border border-slate-700/80 rounded-xl px-3.5 py-1.5 flex items-center space-x-2 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-mono tracking-wider text-slate-400">
              Colorfulness Score
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-sm font-bold font-mono text-cyan-400">
                {metrics ? `${metrics.colorfulnessScore}` : "--"}
              </span>
              <span className="text-[10px] text-slate-400 font-mono">H&S Index</span>
            </div>
          </div>
        </div>

        {/* Resolution */}
        <div className="bg-slate-800/80 border border-slate-700/80 rounded-xl px-3.5 py-1.5 flex items-center space-x-2 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-mono tracking-wider text-slate-400">
              Native Matrix
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-sm font-bold font-mono text-emerald-400">
                {metrics ? metrics.resolution : "600x600"}
              </span>
            </div>
          </div>
        </div>

        {/* Latency */}
        <div className="bg-slate-800/80 border border-slate-700/80 rounded-xl px-3.5 py-1.5 flex items-center space-x-2 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[9px] uppercase font-mono tracking-wider text-slate-400">
              cGAN Synthesis
            </span>
            <div className="flex items-baseline space-x-1">
              {isProcessing ? (
                <span className="text-xs font-mono text-violet-400 animate-pulse">Colorizing...</span>
              ) : (
                <>
                  <span className="text-sm font-bold font-mono text-slate-200">
                    {metrics ? `${metrics.inferenceTimeMs}` : "--"}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">ms</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* 1-Click High-Res Download Button */}
        <button
          onClick={onDownload}
          disabled={!hasColorizedResult}
          className={`py-2 px-3.5 rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-md ${
            hasColorizedResult
              ? "bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white shadow-purple-500/25"
              : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/60"
          }`}
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export HD</span>
        </button>
      </div>
    </header>
  );
};
