import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChromaGAN | AI Black & White Photo Colorizer & Vintage Restorer",
  description:
    "Production Conditional Generative Adversarial Network (cGAN / Pix2Pix) studio for photo colorization in CIELAB color space.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased selection:bg-violet-500/30 selection:text-violet-200">
        {children}
      </body>
    </html>
  );
}
