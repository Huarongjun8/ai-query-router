"use client";

import { useEffect, useRef, useState } from "react";

interface ScoreBarProps {
  value: number;
  max?: number;
  color?: "teal" | "amber" | "red" | "auto";
  label?: string;
  showValue?: boolean;
  height?: string;
}

function resolveColor(color: ScoreBarProps["color"], value: number, max: number): string {
  if (color === "teal") return "#1D9E75";
  if (color === "amber") return "#E4A84B";
  if (color === "red") return "#E24B4A";

  // auto: green → amber → red based on score
  const pct = value / max;
  if (pct <= 0.4) return "#1D9E75";
  if (pct <= 0.7) return "#E4A84B";
  return "#E24B4A";
}

export default function ScoreBar({
  value,
  max = 10,
  color = "auto",
  label,
  showValue = true,
  height = "h-2",
}: ScoreBarProps) {
  const [mounted, setMounted] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const resolvedColor = resolveColor(color, value, max);

  useEffect(() => {
    // Small delay so the transition fires after paint
    const timer = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="w-full">
      {(label || showValue) && (
        <div className="flex justify-between items-center mb-1">
          {label && (
            <span className="text-xs text-ink-secondary font-medium uppercase tracking-wide">
              {label}
            </span>
          )}
          {showValue && (
            <span
              className="text-xs font-mono font-medium ml-auto"
              style={{ color: resolvedColor }}
            >
              {value}/{max}
            </span>
          )}
        </div>
      )}
      <div
        className={`w-full ${height} rounded-full overflow-hidden`}
        style={{ backgroundColor: "#1A2236" }}
      >
        <div
          ref={barRef}
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: mounted ? `${pct}%` : "0%",
            backgroundColor: resolvedColor,
            boxShadow: `0 0 8px ${resolvedColor}66`,
          }}
        />
      </div>
    </div>
  );
}
