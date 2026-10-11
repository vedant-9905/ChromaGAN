"use client";

import React from "react";
import { Activity, Cpu, Database, Flame, Zap, CheckCircle2, AlertTriangle, ShieldCheck } from "lucide-react";
import { MetricData, SystemHealth } from "@/types";

interface HeaderProps {
  systemHealth: SystemHealth;
  metrics: MetricData | null;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  systemHealth,
  metrics,
  isProcessing,
}) => {
  return (
    <header className="h-16 border-b border-pacs-800 bg-pacs-900/90 backdrop-blur-md px-6 flex items-center justify-between z-30 sticky top-0">
      {/* Brand & System Status */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-teal-400 flex items-center justify-center shadow-glow-sm">
            <Activity className="w-5 h-5 text-pacs-950 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-base tracking-wider text-white">
                RadSwin<span className="text-cyan-400 font-extrabold">-WGAN</span>
              </span>
              <span className="text-[10px] font-mono uppercase bg-pacs-800 text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-500/30">
                v1.0.4-PROD
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">
              LDCT Generative Adversarial Restoration Workstation
            </p>
          </div>
        </div>

        {/* Vertical Divider */}
        <div className="h-7 w-[1px] bg-pacs-700/60 mx-1" />

        {/* Backend & Runtime Status Badges */}
        <div className="flex items-center space-x-2">
          {/* Status Indicator */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              systemHealth.status === "ok"
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : systemHealth.status === "degraded"
                ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                : "bg-rose-500/10 text-rose-400 border-rose-500/30"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                systemHealth.status === "ok"
                  ? "bg-emerald-400 animate-pulse"
                  : systemHealth.status === "degraded"
                  ? "bg-amber-400"
                  : "bg-rose-400"
              }`}
            />
            <span className="capitalize">
              {systemHealth.status === "ok"
                ? "WGAN-GP Online"
                : systemHealth.status === "degraded"
                ? "Fallback Ready"
                : "Offline"}
            </span>
          </div>

          {/* Active Device Badge */}
          <div className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-pacs-800/80 border border-pacs-700 text-slate-300 text-xs font-mono">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="uppercase text-[11px] font-semibold">
              {systemHealth.gpuName ? systemHealth.gpuName : systemHealth.device}
            </span>
          </div>
        </div>
      </div>

      {/* Quantitative Metric Badges (Real-Time HUD) */}
      <div className="flex items-center space-x-3">
        {/* PSNR */}
        <div className="bg-pacs-850/90 border border-pacs-700/80 rounded-lg px-3 py-1.5 flex items-center space-x-2.5 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Peak SNR (PSNR)
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-sm font-bold font-mono text-cyan-400">
                {metrics ? `${metrics.psnr.toFixed(2)}` : "--"}
              </span>
              <span className="text-[10px] text-slate-400 font-mono">dB</span>
            </div>
          </div>
        </div>

        {/* SSIM */}
        <div className="bg-pacs-850/90 border border-pacs-700/80 rounded-lg px-3 py-1.5 flex items-center space-x-2.5 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Structural SSIM
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-sm font-bold font-mono text-emerald-400">
                {metrics ? `${metrics.ssim.toFixed(4)}` : "--"}
              </span>
              <span className="text-[10px] text-slate-400 font-mono">/ 1.0</span>
            </div>
          </div>
        </div>

        {/* Reconstruction Fidelity */}
        <div className="bg-pacs-850/90 border border-pacs-700/80 rounded-lg px-3 py-1.5 flex items-center space-x-2.5 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Reconstruction Fidelity
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-sm font-bold font-mono text-teal-300">
                {metrics ? `${metrics.reconstructionFidelity.toFixed(1)}%` : "--%"}
              </span>
            </div>
          </div>
        </div>

        {/* Latency */}
        <div className="bg-pacs-850/90 border border-pacs-700/80 rounded-lg px-3 py-1.5 flex items-center space-x-2.5 shadow-sm">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Inference Latency
            </span>
            <div className="flex items-baseline space-x-1">
              {isProcessing ? (
                <span className="text-xs font-mono text-amber-400 animate-pulse">Running...</span>
              ) : (
                <>
                  <span className="text-sm font-bold font-mono text-slate-200">
                    {metrics ? `${metrics.inferenceTimeMs.toFixed(1)}` : "--"}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">ms</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
