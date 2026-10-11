import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        pacs: {
          950: "#06090e",
          900: "#0b101b",
          850: "#101726",
          800: "#172134",
          700: "#22304a",
          600: "#334566",
          500: "#4b628b",
          400: "#7189b3",
          300: "#9fb2d3",
          200: "#cfdbeb",
          100: "#eaf0f9",
        },
        cyan: {
          450: "#00c4e6",
          500: "#06b6d4",
          600: "#0891b2",
        },
        accent: {
          glow: "rgba(6, 182, 212, 0.15)",
          border: "rgba(34, 211, 238, 0.25)",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 20px -5px rgba(6, 182, 212, 0.3)",
        "glow-sm": "0 0 10px -2px rgba(6, 182, 212, 0.25)",
        card: "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
      },
    },
  },
  plugins: [],
};

export default config;
