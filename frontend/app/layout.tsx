import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RadSwin-WGAN | Radiologist Generative Denoising Workstation",
  description:
    "End-to-End Deep Learning Generative LDCT Restoration Workstation powered by ResUNet WGAN-GP, MS-SSIM, and Perceptual Loss.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-pacs-950 text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
