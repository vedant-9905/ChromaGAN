import { SamplePresetKey } from "@/types";

/**
 * Procedurally draws rich, realistic vintage B&W photographic scenes
 * onto a canvas for instant zero-configuration demonstration.
 */
export function generateVintagePhoto(
  preset: SamplePresetKey,
  asColor: boolean = false,
  width: number = 600,
  height: number = 600
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        throw new Error("Canvas context unavailable");
      }

      if (preset === "vintage_car") {
        // --- 1930s Classic Roadster ---
        // Sky / Background
        const sky = ctx.createLinearGradient(0, 0, 0, height * 0.6);
        sky.addColorStop(0, asColor ? "#4a7bb0" : "#555555");
        sky.addColorStop(1, asColor ? "#d2e3f5" : "#cccccc");
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, width, height * 0.6);

        // Distant hills & trees
        ctx.fillStyle = asColor ? "#2d5a37" : "#3b3b3b";
        ctx.beginPath();
        ctx.arc(width * 0.25, height * 0.65, width * 0.35, 0, Math.PI, true);
        ctx.arc(width * 0.75, height * 0.65, width * 0.4, 0, Math.PI, true);
        ctx.fill();

        // Asphalt / Gravel road
        const road = ctx.createLinearGradient(0, height * 0.55, 0, height);
        road.addColorStop(0, asColor ? "#595048" : "#444444");
        road.addColorStop(1, asColor ? "#332d27" : "#222222");
        ctx.fillStyle = road;
        ctx.fillRect(0, height * 0.55, width, height * 0.45);

        // Car Body (Classic Cherry Red or Vintage Slate)
        ctx.fillStyle = asColor ? "#b91c1c" : "#3a3a3a";
        ctx.beginPath();
        ctx.roundRect(width * 0.15, height * 0.42, width * 0.7, height * 0.22, [25, 60, 10, 10]);
        ctx.fill();

        // Car Cabin / Windshield
        ctx.fillStyle = asColor ? "#1e293b" : "#222222";
        ctx.beginPath();
        ctx.roundRect(width * 0.35, height * 0.32, width * 0.32, height * 0.15, [18, 18, 0, 0]);
        ctx.fill();

        // Windshield Glass
        ctx.fillStyle = asColor ? "#93c5fd" : "#888888";
        ctx.beginPath();
        ctx.moveTo(width * 0.37, height * 0.42);
        ctx.lineTo(width * 0.45, height * 0.34);
        ctx.lineTo(width * 0.62, height * 0.34);
        ctx.lineTo(width * 0.64, height * 0.42);
        ctx.closePath();
        ctx.fill();

        // Chrome details & radiator grille
        ctx.fillStyle = asColor ? "#f1f5f9" : "#e5e5e5";
        ctx.fillRect(width * 0.78, height * 0.44, width * 0.05, height * 0.14);

        // Headlight glow
        ctx.fillStyle = asColor ? "#fef08a" : "#f5f5f5";
        ctx.beginPath();
        ctx.arc(width * 0.82, height * 0.46, 16, 0, 2 * Math.PI);
        ctx.fill();

        // Wheels
        ctx.fillStyle = asColor ? "#0f172a" : "#111111";
        ctx.beginPath();
        ctx.arc(width * 0.3, height * 0.64, 48, 0, 2 * Math.PI);
        ctx.arc(width * 0.7, height * 0.64, 48, 0, 2 * Math.PI);
        ctx.fill();

        // Chrome hubcaps
        ctx.fillStyle = asColor ? "#e2e8f0" : "#d4d4d4";
        ctx.beginPath();
        ctx.arc(width * 0.3, height * 0.64, 20, 0, 2 * Math.PI);
        ctx.arc(width * 0.7, height * 0.64, 20, 0, 2 * Math.PI);
        ctx.fill();

      } else if (preset === "classic_portrait") {
        // --- 1920s Studio Portrait ---
        // Warm/Moody Vignette Backdrop
        const bgGrad = ctx.createRadialGradient(width * 0.5, height * 0.4, 40, width * 0.5, height * 0.5, width * 0.6);
        bgGrad.addColorStop(0, asColor ? "#5c4033" : "#555555");
        bgGrad.addColorStop(1, asColor ? "#1a120b" : "#181818");
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, width, height);

        // Shoulders / Tweed Coat
        ctx.fillStyle = asColor ? "#334155" : "#303030";
        ctx.beginPath();
        ctx.ellipse(width * 0.5, height * 0.85, width * 0.38, height * 0.25, 0, 0, 2 * Math.PI);
        ctx.fill();

        // Collar / Shirt
        ctx.fillStyle = asColor ? "#f8fafc" : "#e2e2e2";
        ctx.beginPath();
        ctx.moveTo(width * 0.44, height * 0.62);
        ctx.lineTo(width * 0.5, height * 0.72);
        ctx.lineTo(width * 0.56, height * 0.62);
        ctx.closePath();
        ctx.fill();

        // Silk Necktie
        ctx.fillStyle = asColor ? "#991b1b" : "#404040";
        ctx.beginPath();
        ctx.moveTo(width * 0.49, height * 0.72);
        ctx.lineTo(width * 0.51, height * 0.72);
        ctx.lineTo(width * 0.52, height * 0.88);
        ctx.lineTo(width * 0.48, height * 0.88);
        ctx.closePath();
        ctx.fill();

        // Neck
        ctx.fillStyle = asColor ? "#e5a786" : "#8d8d8d";
        ctx.fillRect(width * 0.44, height * 0.52, width * 0.12, height * 0.15);

        // Head / Face
        ctx.fillStyle = asColor ? "#f6bc9a" : "#9e9e9e";
        ctx.beginPath();
        ctx.ellipse(width * 0.5, height * 0.42, width * 0.18, height * 0.22, 0, 0, 2 * Math.PI);
        ctx.fill();

        // Cheeks & lighting
        if (asColor) {
          ctx.fillStyle = "rgba(225, 90, 80, 0.2)";
          ctx.beginPath();
          ctx.arc(width * 0.42, height * 0.45, 24, 0, 2 * Math.PI);
          ctx.arc(width * 0.58, height * 0.45, 24, 0, 2 * Math.PI);
          ctx.fill();
        }

        // Eyes
        ctx.fillStyle = asColor ? "#3b2314" : "#222222";
        ctx.beginPath();
        ctx.arc(width * 0.43, height * 0.40, 7, 0, 2 * Math.PI);
        ctx.arc(width * 0.57, height * 0.40, 7, 0, 2 * Math.PI);
        ctx.fill();

        // Lips
        ctx.fillStyle = asColor ? "#b94d4d" : "#555555";
        ctx.beginPath();
        ctx.ellipse(width * 0.5, height * 0.52, 18, 7, 0, 0, 2 * Math.PI);
        ctx.fill();

        // Classic Wavy Hair / Hat
        ctx.fillStyle = asColor ? "#261c14" : "#1a1a1a";
        ctx.beginPath();
        ctx.arc(width * 0.5, height * 0.32, width * 0.20, Math.PI, 0, false);
        ctx.ellipse(width * 0.33, height * 0.40, 25, 60, -0.2, 0, 2 * Math.PI);
        ctx.ellipse(width * 0.67, height * 0.40, 25, 60, 0.2, 0, 2 * Math.PI);
        ctx.fill();

      } else if (preset === "old_city") {
        // --- 1910s Historic City Street ---
        // Sky
        const sky = ctx.createLinearGradient(0, 0, 0, height * 0.45);
        sky.addColorStop(0, asColor ? "#fdba74" : "#a8a8a8"); // sunset golden sky
        sky.addColorStop(1, asColor ? "#fed7aa" : "#cccccc");
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, width, height * 0.45);

        // Cobblestone ground
        ctx.fillStyle = asColor ? "#475569" : "#3a3a3a";
        ctx.fillRect(0, height * 0.45, width, height * 0.55);

        // Left Victorian Brick Building
        ctx.fillStyle = asColor ? "#854d0e" : "#4a4a4a";
        ctx.fillRect(0, height * 0.15, width * 0.35, height * 0.65);

        // Right Stone Building
        ctx.fillStyle = asColor ? "#78716c" : "#545454";
        ctx.fillRect(width * 0.65, height * 0.10, width * 0.35, height * 0.70);

        // Windows with warm lantern light
        ctx.fillStyle = asColor ? "#fef08a" : "#eaeaea";
        for (let row = 0; row < 4; row++) {
          ctx.fillRect(width * 0.08, height * 0.22 + row * 60, 36, 40);
          ctx.fillRect(width * 0.20, height * 0.22 + row * 60, 36, 40);
          ctx.fillRect(width * 0.72, height * 0.18 + row * 65, 38, 42);
          ctx.fillRect(width * 0.85, height * 0.18 + row * 65, 38, 42);
        }

        // Vintage Cast-Iron Gas Lamp
        ctx.strokeStyle = asColor ? "#0f172a" : "#1a1a1a";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(width * 0.42, height * 0.72);
        ctx.lineTo(width * 0.42, height * 0.48);
        ctx.stroke();

        ctx.fillStyle = asColor ? "#f59e0b" : "#f0f0f0";
        ctx.beginPath();
        ctx.arc(width * 0.42, height * 0.47, 10, 0, 2 * Math.PI);
        ctx.fill();

      } else {
        // --- Landscape (National Park Yosemite style) ---
        // Sky
        const sky = ctx.createLinearGradient(0, 0, 0, height * 0.5);
        sky.addColorStop(0, asColor ? "#0284c7" : "#444444");
        sky.addColorStop(1, asColor ? "#bae6fd" : "#c2c2c2");
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, width, height * 0.5);

        // Sun / Morning glow
        ctx.fillStyle = asColor ? "#fffbeb" : "#fefefe";
        ctx.beginPath();
        ctx.arc(width * 0.75, height * 0.20, 36, 0, 2 * Math.PI);
        ctx.fill();

        // Mountain Peaks (Granite rock)
        ctx.fillStyle = asColor ? "#475569" : "#555555";
        ctx.beginPath();
        ctx.moveTo(0, height * 0.5);
        ctx.lineTo(width * 0.3, height * 0.18);
        ctx.lineTo(width * 0.55, height * 0.42);
        ctx.lineTo(width * 0.8, height * 0.15);
        ctx.lineTo(width, height * 0.5);
        ctx.closePath();
        ctx.fill();

        // Snow on peaks
        ctx.fillStyle = asColor ? "#ffffff" : "#eeeeee";
        ctx.beginPath();
        ctx.moveTo(width * 0.3, height * 0.18);
        ctx.lineTo(width * 0.25, height * 0.26);
        ctx.lineTo(width * 0.35, height * 0.26);
        ctx.closePath();
        ctx.fill();

        // Alpine Lake Reflection
        const lake = ctx.createLinearGradient(0, height * 0.5, 0, height);
        lake.addColorStop(0, asColor ? "#0369a1" : "#333333");
        lake.addColorStop(1, asColor ? "#075985" : "#222222");
        ctx.fillStyle = lake;
        ctx.fillRect(0, height * 0.5, width, height * 0.5);

        // Pine Trees on shoreline
        ctx.fillStyle = asColor ? "#14532d" : "#2a2a2a";
        for (let t = 0; t < 12; t++) {
          const tx = t * (width / 11);
          ctx.beginPath();
          ctx.moveTo(tx, height * 0.52);
          ctx.lineTo(tx + 18, height * 0.40);
          ctx.lineTo(tx + 36, height * 0.52);
          ctx.closePath();
          ctx.fill();
        }
      }

      // If generating vintage B&W, add authentic film grain & desaturation
      if (!asColor) {
        const imgData = ctx.getImageData(0, 0, width, height);
        const data = imgData.data;

        for (let i = 0; i < data.length; i += 4) {
          // Standard ITU-R BT.601 luma formula
          const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          // Film grain
          const grain = (Math.random() - 0.5) * 16;
          const finalVal = Math.min(255, Math.max(0, gray + grain));

          data[i] = finalVal;
          data[i + 1] = finalVal;
          data[i + 2] = finalVal;
        }

        ctx.putImageData(imgData, 0, 0);
      }

      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Canvas blob export failed"));
        }
      }, "image/png");
    } catch (err) {
      reject(err);
    }
  });
}
