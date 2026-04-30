import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0B0F1A",
          mid: "#111827",
          high: "#1A2236",
        },
        "accent-teal": "#1D9E75",
        "accent-amber": "#E4A84B",
        "accent-red": "#E24B4A",
        ink: {
          primary: "#F0F4FF",
          secondary: "#9BA8C0",
          muted: "#5C6882",
        },
      },
      fontFamily: {
        sans: ["DM Sans", "sans-serif"],
        mono: ["DM Mono", "monospace"],
      },
      animation: {
        "fade-up": "fadeUp 0.4s ease-out forwards",
        "score-bar": "scoreBar 0.8s ease-out forwards",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        scoreBar: {
          "0%": { width: "0%" },
          "100%": { width: "var(--score-width)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
