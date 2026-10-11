import { PresetArtifactType } from "@/types";

/**
 * Procedurally generates an anatomically realistic axial thoracic/lung CT slice
 * on an HTML5 canvas and applies configurable low-dose CT noise / artifacts.
 */
export function generateSyntheticCTSlice(
  type: PresetArtifactType = "normal",
  size: number = 512
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        throw new Error("Unable to obtain 2D canvas context");
      }

      // Background air (-1000 HU -> black)
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, size, size);

      const cx = size / 2;
      const cy = size / 2;

      // 1. Outer Thoracic Body Outline (Soft Tissue: ~40 HU)
      ctx.beginPath();
      ctx.ellipse(cx, cy + 10, size * 0.44, size * 0.38, 0, 0, 2 * Math.PI);
      ctx.fillStyle = "#3a3a3a";
      ctx.fill();

      // 2. Subcutaneous Fat layer (slightly lower attenuation)
      ctx.beginPath();
      ctx.ellipse(cx, cy + 10, size * 0.40, size * 0.35, 0, 0, 2 * Math.PI);
      ctx.fillStyle = "#2c2c2c";
      ctx.fill();

      // 3. Thoracic Ribs / Spine / Sternum (Bone: ~400-1000 HU -> Bright White)
      ctx.fillStyle = "#e0e0e0";
      // Vertebral Body (Spine at posterior)
      ctx.beginPath();
      ctx.arc(cx, cy + size * 0.28, size * 0.055, 0, 2 * Math.PI);
      ctx.fill();
      // Spinal canal (lower density inside spine)
      ctx.beginPath();
      ctx.arc(cx, cy + size * 0.28, size * 0.02, 0, 2 * Math.PI);
      ctx.fillStyle = "#333333";
      ctx.fill();

      // Sternum (anterior)
      ctx.fillStyle = "#dedede";
      ctx.beginPath();
      ctx.ellipse(cx, cy - size * 0.31, size * 0.04, size * 0.015, 0, 0, 2 * Math.PI);
      ctx.fill();

      // Rib cage pairs
      const ribCount = 7;
      for (let i = 0; i < ribCount; i++) {
        const angle = (Math.PI / (ribCount + 1)) * (i + 1);
        const rx = Math.sin(angle) * (size * 0.38);
        const ry = (Math.cos(angle) * (size * 0.30)) + 10;
        
        // Left rib
        ctx.beginPath();
        ctx.ellipse(cx - rx, cy + ry, size * 0.022, size * 0.012, angle * 0.3, 0, 2 * Math.PI);
        ctx.fill();

        // Right rib
        ctx.beginPath();
        ctx.ellipse(cx + rx, cy + ry, size * 0.022, size * 0.012, -angle * 0.3, 0, 2 * Math.PI);
        ctx.fill();
      }

      // 4. Mediastinum & Cardiac Silhouette (Central Soft Tissue)
      ctx.fillStyle = "#505050";
      ctx.beginPath();
      ctx.ellipse(cx + 8, cy - 10, size * 0.16, size * 0.22, -0.15, 0, 2 * Math.PI);
      ctx.fill();

      // Trachea / Bronchial tree lumen (air -> dark)
      ctx.fillStyle = "#0a0a0a";
      ctx.beginPath();
      ctx.arc(cx, cy - size * 0.08, size * 0.025, 0, 2 * Math.PI);
      ctx.fill();

      // 5. Lung Fields (Left & Right Lung - Parenchyma: ~-800 to -600 HU -> Very Dark Gray)
      // Right Lung (anatomical right = viewer's left)
      ctx.fillStyle = "#121212";
      ctx.beginPath();
      ctx.ellipse(cx - size * 0.19, cy + 5, size * 0.14, size * 0.26, 0.1, 0, 2 * Math.PI);
      ctx.fill();

      // Left Lung (anatomical left = viewer's right, slightly indented for cardiac notch)
      ctx.beginPath();
      ctx.ellipse(cx + size * 0.20, cy + 18, size * 0.13, size * 0.24, -0.18, 0, 2 * Math.PI);
      ctx.fill();

      // 6. Bronchovascular Markings / Pulmonary Vessels (Light gray branching structures)
      ctx.strokeStyle = "#4d4d4d";
      ctx.lineWidth = 2.5;

      const drawVessels = (startX: number, startY: number, dirX: number) => {
        for (let j = 0; j < 6; j++) {
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          const endX = startX + dirX * (30 + Math.random() * 55);
          const endY = startY + (Math.random() - 0.5) * 80;
          ctx.quadraticCurveTo(
            startX + dirX * 20,
            startY + (Math.random() - 0.5) * 30,
            endX,
            endY
          );
          ctx.stroke();

          // Small pulmonary nodule or cross-section
          ctx.beginPath();
          ctx.arc(endX, endY, 2 + Math.random() * 2, 0, 2 * Math.PI);
          ctx.fillStyle = "#666666";
          ctx.fill();
        }
      };

      drawVessels(cx - size * 0.12, cy - 10, -1);
      drawVessels(cx + size * 0.12, cy - 5, 1);

      // 7. Apply Low-Dose CT Noise and Physical Degradation Artifacts
      const imgData = ctx.getImageData(0, 0, size, size);
      const data = imgData.data;

      const noiseLevel = type === "high_noise" ? 55 : type === "streak_artifact" ? 75 : 12;

      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = (y * size + x) * 4;

          // Only add noise inside body or air background with photon starvation
          let noise = (Math.random() - 0.5) * noiseLevel;

          // Add Poisson-like quantum mottle
          const val = data[idx];
          if (val > 10) {
            const poisson = (Math.random() - 0.5) * Math.sqrt(val) * (type === "normal" ? 0.8 : 3.2);
            noise += poisson;
          }

          // Photon starvation streak artifacts (directional high-attenuation shadows)
          if (type === "streak_artifact") {
            const angle1 = (x * 0.866 + y * 0.5);
            const streak1 = Math.sin(angle1 * 0.18) * 28 * Math.exp(-Math.abs(y - (cy + size * 0.28)) / 80);
            const streak2 = Math.cos((x - y) * 0.22) * 20 * Math.exp(-Math.abs(y - cy) / 100);
            noise += (streak1 + streak2);
          }

          const noisyVal = Math.min(255, Math.max(0, data[idx] + noise));
          data[idx] = noisyVal;     // R
          data[idx + 1] = noisyVal; // G
          data[idx + 2] = noisyVal; // B
        }
      }

      ctx.putImageData(imgData, 0, 0);

      // Return Blob
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Canvas blob generation failed"));
        }
      }, "image/png");
    } catch (err) {
      reject(err);
    }
  });
}
