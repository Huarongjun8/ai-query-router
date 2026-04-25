"use client";

import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  ZoomableGroup,
} from "react-simple-maps";
import type { BriefResponse } from "@/lib/api";

// ── Constants ─────────────────────────────────────────────────────────────────

const GEO_URL =
  "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://web-production-ae1f5.up.railway.app";

const POLL_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

// ── Types ─────────────────────────────────────────────────────────────────────

type SignalType = "risk" | "opportunity" | "mixed";

interface RegionSignal {
  text: string;
  source: string;
}

interface Region {
  id: string;
  name: string;
  countryIds: string[]; // ISO 3166-1 numeric strings (world-atlas IDs)
  signalType: SignalType;
  riskScore: number; // 0–10
  signals: RegionSignal[];
  summary: string;
  briefId?: string;
  lastUpdated: string;
  isLive?: boolean;
  isPulsing?: boolean; // rapidly changing score
}

// ── Seed data — 6 pre-loaded high-signal regions ─────────────────────────────

const SEED_REGIONS: Region[] = [
  {
    id: "taiwan-strait",
    name: "Taiwan Strait",
    countryIds: ["156"], // China (Taiwan not separate in 110m topojson)
    signalType: "risk",
    riskScore: 8.5,
    isPulsing: true,
    signals: [
      {
        text: "PLA warships conducting rare transits near southwest Japan",
        source: "SCMP",
      },
      {
        text: "Type 076 drone carrier deployed for South China Sea training drills",
        source: "Defense News",
      },
      {
        text: "Taiwan President Lai's Africa trip blocked via Beijing diplomatic pressure",
        source: "BBC",
      },
    ],
    summary:
      "Elevated cross-strait tension driven by accelerating PLA power projection and Beijing's systematic diplomatic isolation campaign against Taiwan. China's 2026 defense budget increase to 1.94 trillion yuan signals enhanced military readiness. The window of maximum risk appears to be 2027–2029, when PLA amphibious capabilities peak relative to Taiwan's defensive development.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "russia-ukraine",
    name: "Russia–Ukraine",
    countryIds: ["643", "804"], // Russia, Ukraine
    signalType: "risk",
    riskScore: 7.8,
    isPulsing: true,
    signals: [
      {
        text: "Russia could be ready for NATO conflict within one year, Dutch intel warns",
        source: "Defense News",
      },
      {
        text: "EU approves €90bn Ukraine loan; new Russia sanctions package passed",
        source: "Al Jazeera",
      },
      {
        text: "Germany unveils strategy to become Europe's strongest military by 2039",
        source: "Defense News",
      },
    ],
    summary:
      "Sustained high-intensity conflict with increasing NATO involvement risk. European defense spending acceleration signals a long-term structural shift in continental security posture. EU financial commitment to Ukraine is hardening despite domestic political pressures in member states.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "us-china-tech",
    name: "US–China Tech War",
    countryIds: ["840"], // United States
    signalType: "risk",
    riskScore: 7.2,
    isPulsing: false,
    signals: [
      {
        text: "China bought zero H200 chips — Lutnick cites 'delicate balance' with Xi",
        source: "SCMP",
      },
      {
        text: "Beijing's security emphasis weighing on US firms' China optimism — AmCham",
        source: "SCMP",
      },
      {
        text: "Chinese AI development constrained: 'a snake eating its own tail'",
        source: "Defense News",
      },
    ],
    summary:
      "Technology decoupling accelerating with semiconductor export controls creating a bifurcated global tech stack. US firms face growing operational uncertainty in China. Long-term structural separation of US-China technology ecosystems now appears inevitable regardless of diplomatic outcomes.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "iran",
    name: "Iran / Strait of Hormuz",
    countryIds: ["364"], // Iran
    signalType: "risk",
    riskScore: 6.5,
    isPulsing: false,
    signals: [
      {
        text: "Iran–US negotiations stalled; Trump madman diplomacy not working",
        source: "Defense News",
      },
      {
        text: "Russia–Iran nexus reshaping global conflict dynamics",
        source: "OilPrice",
      },
      {
        text: "Pakistan mediating between Iran and US via active diplomatic back-channel",
        source: "The Diplomat",
      },
    ],
    summary:
      "Nuclear negotiations remain deadlocked with low probability of near-term breakthrough. Strait of Hormuz transit risk elevated by Iran-Russia coordination. The Pakistan back-channel represents the only active diplomatic pathway — its collapse would significantly raise military confrontation probability.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "korean-peninsula",
    name: "Korean Peninsula",
    countryIds: ["408", "410"], // North Korea, South Korea
    signalType: "mixed",
    riskScore: 5.5,
    isPulsing: false,
    signals: [
      {
        text: "North Korea–Russia military cooperation deepening amid Ukraine conflict",
        source: "Nikkei Asia",
      },
      {
        text: "US commander warns against 'starving the chicken' on Korea defense",
        source: "Defense News",
      },
      {
        text: "South Korea Q1 GDP returns to growth on semiconductor export strength",
        source: "Nikkei Asia",
      },
    ],
    summary:
      "Peninsula stability maintained by mutual deterrence, but the North Korea–Russia arms nexus introduces new escalation pathways not covered by existing deterrence architecture. South Korean economic resilience through semiconductor exports provides fiscal buffer for defence investment.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "india-supply-chain",
    name: "India–Indo-Pacific",
    countryIds: ["356"], // India
    signalType: "opportunity",
    riskScore: 3.5,
    isPulsing: false,
    signals: [
      {
        text: "India rare earth supply chains emerging as strategic lever in Indo-Pacific",
        source: "The Diplomat",
      },
      {
        text: "Japan–India cooperation on resource supply chains in new Indo-Pacific plan",
        source: "Nikkei Asia",
      },
      {
        text: "Vietnam–South Korea cooperation on supply chains and nuclear energy agreed",
        source: "The Diplomat",
      },
    ],
    summary:
      "India's emerging role as an alternative technology supply chain anchor creates significant opportunity for financial services firms with Asia-Pacific exposure. Supply chain diversification away from China-concentrated production is accelerating across semiconductors, rare earths, and critical minerals.",
    lastUpdated: new Date().toISOString(),
  },
];

// ── Color helpers ─────────────────────────────────────────────────────────────

function riskFill(score: number, type: SignalType): string {
  const t = Math.max(0, Math.min(10, score)) / 10;

  if (type === "opportunity") {
    // light mint (#d1fae5) → deep green (#064e3b)
    const r = Math.round(209 - 209 * t + 6 * t);
    const g = Math.round(250 - 250 * t + 78 * t);
    const b = Math.round(229 - 229 * t + 59 * t);
    return `rgb(${r},${g},${b})`;
  }

  if (type === "mixed") {
    // light amber (#fef3c7) → deep amber (#78350f)
    const r = Math.round(254 - 254 * t + 120 * t);
    const g = Math.round(243 - 243 * t + 53 * t);
    const b = Math.round(199 - 199 * t + 15 * t);
    return `rgb(${r},${g},${b})`;
  }

  // risk: light pink (#fee2e2) → deep crimson (#450a0a)
  const r = Math.round(254 - 254 * t + 69 * t);
  const g = Math.round(226 - 226 * t + 10 * t);
  const b = Math.round(226 - 226 * t + 10 * t);
  return `rgb(${r},${g},${b})`;
}

function hoverFill(score: number, type: SignalType): string {
  return riskFill(Math.min(10, score + 1.5), type);
}

function urgencyLabel(score: number): string {
  if (score >= 8) return "CRITICAL";
  if (score >= 6.5) return "HIGH";
  if (score >= 4.5) return "MEDIUM";
  return "LOW";
}

function urgencyColor(score: number): string {
  if (score >= 8) return "#e24b4a";
  if (score >= 6.5) return "#e4a84b";
  if (score >= 4.5) return "#9ba8c0";
  return "#1d9e75";
}

function signalTypeColor(type: SignalType): string {
  if (type === "risk") return "#e24b4a";
  if (type === "opportunity") return "#1d9e75";
  return "#e4a84b";
}

// ── Main component ────────────────────────────────────────────────────────────

export default function GeopoliticalMap() {
  const [regions, setRegions] = useState<Region[]>(SEED_REGIONS);
  const [selected, setSelected] = useState<Region | null>(null);
  const [tooltip, setTooltip] = useState<{
    region: Region;
    x: number;
    y: number;
  } | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [lastPollTime, setLastPollTime] = useState<Date | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Build country-ID → region lookup
  const countryMap = useMemo(
    () =>
      Object.fromEntries(
        regions.flatMap((r) => r.countryIds.map((id) => [id, r]))
      ),
    [regions]
  );

  // ── Live poll ──────────────────────────────────────────────────────────────

  const pollLive = useCallback(async () => {
    setIsPolling(true);
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query:
            "Analyse current Taiwan Strait tensions, US-China semiconductor decoupling, and Russia-Ukraine escalation risk signals",
        }),
      });
      if (!res.ok) return;
      const brief: BriefResponse = await res.json();
      if (brief.status !== "complete") return;
      const a = brief.risk_assessments?.[0];
      if (!a) return;
      const newScore = a.risk_score ?? 8.5;

      setRegions((prev) =>
        prev.map((r) => {
          if (r.id === "taiwan-strait") {
            return {
              ...r,
              riskScore: newScore,
              summary: a.summary ?? r.summary,
              signals:
                a.sources?.slice(0, 3).map((src) => {
                  let host = src;
                  try {
                    host = new URL(src).hostname.replace(/^www\./, "");
                  } catch {}
                  return { text: a.title ?? "Live signal", source: host };
                }) ?? r.signals,
              lastUpdated: new Date().toISOString(),
              isLive: true,
              briefId: brief.brief_id,
            };
          }
          if (r.id === "us-china-tech") {
            return {
              ...r,
              riskScore: Math.max(4, newScore - 1.5),
              lastUpdated: new Date().toISOString(),
              isLive: true,
            };
          }
          return r;
        })
      );
      setLastPollTime(new Date());
    } catch {
      // non-critical — keep seed data
    } finally {
      setIsPolling(false);
    }
  }, []);

  // Schedule recurring poll
  useEffect(() => {
    pollTimer.current = setTimeout(() => {
      pollLive().then(() => {
        // restart timer after completion
        pollTimer.current = setTimeout(pollLive, POLL_INTERVAL_MS);
      });
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [pollLive]);

  // Close panel on Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      {/* ── Keyframes ─────────────────────────────────────────────────────── */}
      <style>{`
        @keyframes breathe {
          0%, 100% { opacity: 0.75; }
          50%       { opacity: 1; }
        }
        @keyframes live-dot {
          0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(29,158,117,0.5); }
          50%       { opacity: 0.6; box-shadow: 0 0 0 5px rgba(29,158,117,0); }
        }
        @keyframes slide-up {
          from { transform: translateY(100%); opacity: 0; }
          to   { transform: translateY(0);    opacity: 1; }
        }
        @keyframes slide-in-right {
          from { transform: translateX(100%); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        .breathe          { animation: breathe 3s ease-in-out infinite; }
        .live-indicator   { animation: live-dot 2s ease-in-out infinite; }
        .panel-mobile     { animation: slide-up 0.3s ease-out; }
        .panel-desktop    { animation: slide-in-right 0.3s ease-out; }
        .overlay-fade     { animation: fade-in 0.2s ease-out; }
        /* Ocean / base map */
        .rsm-geographies { background: #0d1525; }
      `}</style>

      <div className="min-h-screen flex flex-col bg-[#0b0f1a] overflow-hidden">
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="shrink-0 z-20 border-b border-[#1a2236] px-4 sm:px-6 py-3 flex items-center gap-3 bg-[#0b0f1a]/95 backdrop-blur-sm">
          {/* Brand */}
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md bg-[#1d9e75]/20 border border-[#1d9e75]/40 flex items-center justify-center">
              <span className="text-[#1d9e75] text-[10px] font-bold font-mono tracking-tight">
                AS
              </span>
            </div>
            <span className="text-sm font-semibold text-[#f0f4ff] tracking-tight">
              AgentSafe
            </span>
            <span className="text-[10px] font-mono text-[#5c6882] border border-[#1a2236] rounded px-1.5 py-0.5 hidden sm:inline">
              BETA
            </span>
          </div>

          <div className="h-4 w-px bg-[#1a2236]" />

          {/* Live label */}
          <div className="flex items-center gap-2">
            <span
              className="live-indicator w-2 h-2 rounded-full bg-[#1d9e75] shrink-0"
              aria-label="Live"
            />
            <span className="text-xs text-[#9ba8c0] font-mono tracking-wide hidden sm:inline">
              LIVE GEOPOLITICAL INTELLIGENCE
            </span>
            <span className="text-xs text-[#9ba8c0] font-mono tracking-wide sm:hidden">
              LIVE
            </span>
          </div>

          {/* Right controls */}
          <div className="ml-auto flex items-center gap-3">
            {isPolling && (
              <span className="text-xs font-mono text-[#e4a84b] animate-pulse hidden sm:inline">
                ↻ UPDATING
              </span>
            )}
            {lastPollTime && !isPolling && (
              <span className="text-[10px] font-mono text-[#5c6882] hidden md:inline">
                Live update {lastPollTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            )}
            <button
              onClick={pollLive}
              disabled={isPolling}
              className="text-xs font-mono text-[#1d9e75] border border-[#1d9e75]/30 rounded px-2.5 py-1 hover:bg-[#1d9e75]/10 disabled:opacity-40 transition-colors"
            >
              {isPolling ? "…" : "Refresh"}
            </button>
            <a
              href="/analyze"
              className="text-xs font-mono text-[#9ba8c0] border border-[#1a2236] rounded px-2.5 py-1 hover:border-[#1d9e75]/40 hover:text-[#1d9e75] transition-colors hidden sm:inline-block"
            >
              Query →
            </a>
          </div>
        </header>

        {/* ── Map ─────────────────────────────────────────────────────────── */}
        <div className="relative flex-1 overflow-hidden bg-[#0d1525]">
          <ComposableMap
            projectionConfig={{ scale: 155, center: [15, 10] }}
            width={960}
            height={560}
            style={{ width: "100%", height: "100%" }}
          >
            <ZoomableGroup zoom={1} minZoom={0.6} maxZoom={8}>
              <Geographies geography={GEO_URL}>
                {({ geographies }) =>
                  geographies.map((geo) => {
                    const id = String(geo.id);
                    const region = countryMap[id];
                    const isHighRisk = region && region.riskScore > 7;
                    const fill = region
                      ? riskFill(region.riskScore, region.signalType)
                      : "#1a2236";
                    const hover = region
                      ? hoverFill(region.riskScore, region.signalType)
                      : "#1e2a3d";

                    return (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill={fill}
                        stroke="#0b0f1a"
                        strokeWidth={0.4}
                        className={isHighRisk ? "breathe" : ""}
                        style={{
                          default: { fill, outline: "none" },
                          hover: {
                            fill: hover,
                            outline: "none",
                            cursor: region ? "pointer" : "default",
                          },
                          pressed: { outline: "none" },
                        }}
                        onClick={() => region && setSelected(region)}
                        onMouseEnter={(e: React.MouseEvent) => {
                          if (region)
                            setTooltip({ region, x: e.clientX, y: e.clientY });
                        }}
                        onMouseMove={(e: React.MouseEvent) => {
                          if (region)
                            setTooltip((t) =>
                              t ? { ...t, x: e.clientX, y: e.clientY } : null
                            );
                        }}
                        onMouseLeave={() => setTooltip(null)}
                      />
                    );
                  })
                }
              </Geographies>
            </ZoomableGroup>
          </ComposableMap>

          {/* ── Legend ──────────────────────────────────────────────────── */}
          <div className="absolute bottom-4 left-4 bg-[#0b0f1a]/90 border border-[#1a2236] rounded-xl p-3 backdrop-blur-sm text-[10px] font-mono">
            <p className="text-[#5c6882] uppercase tracking-widest mb-2">
              Heat Signature
            </p>
            <div className="flex flex-col gap-1.5">
              {[
                { swatch: "bg-[#450a0a]", label: "Critical Risk  ≥8.0" },
                { swatch: "bg-[#991b1b]", label: "High Risk      ≥6.5" },
                { swatch: "bg-[#d97706]/80", label: "Mixed / Uncertain" },
                { swatch: "bg-[#064e3b]", label: "Opportunity Signal" },
                { swatch: "bg-[#1a2236]", label: "No Signal" },
              ].map(({ swatch, label }) => (
                <div key={label} className="flex items-center gap-2">
                  <span className={`w-3 h-2.5 rounded-sm ${swatch} shrink-0`} />
                  <span className="text-[#9ba8c0]">{label}</span>
                </div>
              ))}
            </div>
            <p className="text-[#5c6882] mt-2.5">Click region for brief</p>
          </div>

          {/* ── Region score sidebar ─────────────────────────────────────── */}
          <div className="absolute top-3 right-3 flex flex-col gap-1.5 w-[170px]">
            {regions
              .slice()
              .sort((a, b) => b.riskScore - a.riskScore)
              .map((r) => (
                <button
                  key={r.id}
                  onClick={() => setSelected(r)}
                  className="flex items-center gap-2 bg-[#0b0f1a]/90 border border-[#1a2236] rounded-lg px-2.5 py-2 backdrop-blur-sm hover:border-[#1d9e75]/40 transition-all text-left group"
                >
                  {/* Score dot */}
                  <span
                    className={`w-2 h-2 rounded-full shrink-0 ${
                      r.riskScore > 7 ? "animate-pulse" : ""
                    }`}
                    style={{
                      background: riskFill(r.riskScore, r.signalType),
                    }}
                  />
                  {/* Name */}
                  <span className="text-[11px] text-[#f0f4ff] truncate flex-1 font-medium group-hover:text-[#1d9e75] transition-colors">
                    {r.name}
                  </span>
                  {/* Score */}
                  <span
                    className="text-[10px] font-mono font-bold shrink-0"
                    style={{ color: urgencyColor(r.riskScore) }}
                  >
                    {r.riskScore.toFixed(1)}
                  </span>
                </button>
              ))}
          </div>

          {/* ── Tooltip ─────────────────────────────────────────────────── */}
          {tooltip && (
            <div
              className="fixed z-50 pointer-events-none bg-[#111827]/95 border border-[#1a2236] rounded-lg px-3 py-2 backdrop-blur-sm shadow-xl"
              style={{ left: tooltip.x + 14, top: tooltip.y - 8 }}
            >
              <p className="text-xs font-semibold text-[#f0f4ff]">
                {tooltip.region.name}
              </p>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span
                  className="text-[10px] font-mono font-bold"
                  style={{ color: urgencyColor(tooltip.region.riskScore) }}
                >
                  {urgencyLabel(tooltip.region.riskScore)}
                </span>
                <span className="text-[10px] text-[#5c6882]">·</span>
                <span className="text-[10px] text-[#9ba8c0]">
                  {tooltip.region.riskScore.toFixed(1)}/10
                </span>
              </div>
            </div>
          )}
        </div>

        {/* ── Detail panel ────────────────────────────────────────────────── */}
        {selected && (
          <div className="fixed inset-0 z-30 flex items-end md:items-center justify-end">
            {/* Backdrop */}
            <div
              className="overlay-fade absolute inset-0 bg-black/50"
              onClick={() => setSelected(null)}
            />

            {/* Panel */}
            <div className="panel-mobile md:panel-desktop relative z-10 w-full md:w-[420px] md:mr-4 md:my-4 bg-[#111827] border border-[#1a2236] rounded-t-2xl md:rounded-2xl shadow-2xl max-h-[85dvh] overflow-y-auto">
              {/* Panel header */}
              <div className="flex items-start justify-between p-5 border-b border-[#1a2236] sticky top-0 bg-[#111827] rounded-t-2xl z-10">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="text-[10px] font-mono font-bold tracking-wider"
                      style={{ color: urgencyColor(selected.riskScore) }}
                    >
                      {urgencyLabel(selected.riskScore)}
                    </span>
                    <span
                      className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                      style={{
                        color: signalTypeColor(selected.signalType),
                        borderColor: `${signalTypeColor(selected.signalType)}40`,
                      }}
                    >
                      {selected.signalType.toUpperCase()}
                    </span>
                    {selected.isLive && (
                      <span className="text-[10px] font-mono text-[#1d9e75] border border-[#1d9e75]/30 rounded px-1.5 py-0.5">
                        LIVE
                      </span>
                    )}
                  </div>
                  <h2 className="text-base font-semibold text-[#f0f4ff]">
                    {selected.name}
                  </h2>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-[#5c6882] hover:text-[#f0f4ff] transition-colors text-2xl leading-none mt-0.5 ml-4 shrink-0"
                  aria-label="Close"
                >
                  ×
                </button>
              </div>

              {/* Score metrics */}
              <div className="grid grid-cols-3 gap-3 p-4 border-b border-[#1a2236]">
                <div className="bg-[#0b0f1a] rounded-xl p-3">
                  <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-wider mb-1">
                    Risk Score
                  </p>
                  <p
                    className="text-lg font-bold font-mono"
                    style={{ color: urgencyColor(selected.riskScore) }}
                  >
                    {selected.riskScore.toFixed(1)}
                    <span className="text-xs text-[#5c6882] font-normal">/10</span>
                  </p>
                </div>
                <div className="bg-[#0b0f1a] rounded-xl p-3">
                  <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-wider mb-1">
                    Signal
                  </p>
                  <p
                    className="text-sm font-semibold capitalize"
                    style={{ color: signalTypeColor(selected.signalType) }}
                  >
                    {selected.signalType}
                  </p>
                </div>
                <div className="bg-[#0b0f1a] rounded-xl p-3">
                  <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-wider mb-1">
                    Updated
                  </p>
                  <p className="text-sm text-[#9ba8c0]">
                    {new Date(selected.lastUpdated).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>

              {/* Score bar */}
              <div className="px-4 py-3 border-b border-[#1a2236]">
                <div className="flex justify-between text-[9px] font-mono text-[#5c6882] mb-1.5">
                  <span>0</span>
                  <span className="text-[#9ba8c0]">Risk Intensity</span>
                  <span>10</span>
                </div>
                <div className="h-2 bg-[#1a2236] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${(selected.riskScore / 10) * 100}%`,
                      background: riskFill(selected.riskScore, selected.signalType),
                    }}
                  />
                </div>
              </div>

              {/* Summary */}
              <div className="px-4 py-4 border-b border-[#1a2236]">
                <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-widest mb-2">
                  Assessment
                </p>
                <p className="text-sm text-[#9ba8c0] leading-relaxed">
                  {selected.summary}
                </p>
              </div>

              {/* Top 3 signals */}
              <div className="px-4 py-4 border-b border-[#1a2236]">
                <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-widest mb-3">
                  Contributing Signals
                </p>
                <div className="flex flex-col gap-3">
                  {selected.signals.slice(0, 3).map((sig, i) => (
                    <div key={i} className="flex gap-3">
                      <span className="text-[10px] font-mono text-[#1d9e75] mt-0.5 shrink-0 font-bold">
                        {i + 1}
                      </span>
                      <div>
                        <p className="text-xs text-[#f0f4ff] leading-snug">
                          {sig.text}
                        </p>
                        <p className="text-[10px] text-[#5c6882] font-mono mt-0.5">
                          {sig.source}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Footer */}
              <div className="px-4 py-4 flex items-center justify-between gap-3">
                <p className="text-[10px] text-[#5c6882] font-mono">
                  {selected.isLive
                    ? "Source: AgentSafe live pipeline"
                    : "Demo data · click Fetch for live brief"}
                </p>
                {selected.briefId ? (
                  <a
                    href={`/brief/${selected.briefId}`}
                    className="shrink-0 text-xs text-[#1d9e75] border border-[#1d9e75]/30 rounded-lg px-3 py-1.5 hover:bg-[#1d9e75]/10 transition-colors font-medium"
                  >
                    Full Brief →
                  </a>
                ) : (
                  <button
                    onClick={() => {
                      setSelected(null);
                      pollLive();
                    }}
                    disabled={isPolling}
                    className="shrink-0 text-xs text-[#1d9e75] border border-[#1d9e75]/30 rounded-lg px-3 py-1.5 hover:bg-[#1d9e75]/10 disabled:opacity-40 transition-colors font-medium"
                  >
                    {isPolling ? "Fetching…" : "Fetch Live Brief →"}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
