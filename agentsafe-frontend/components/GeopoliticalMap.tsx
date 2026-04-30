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
  mineralTags?: string[]; // critical minerals exposure
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
  // ── Africa & LatAm Critical Minerals ────────────────────────────────────────
  {
    id: "drc-cobalt",
    name: "DRC — Cobalt Belt",
    countryIds: ["180"], // Democratic Republic of Congo
    signalType: "risk",
    riskScore: 7.4,
    isPulsing: true,
    mineralTags: ["cobalt", "copper"],
    signals: [
      { text: "CMOC and Glencore competing for Tenke Fungurume expansion rights", source: "Mining.com" },
      { text: "M23 advance toward Goma disrupting eastern DRC supply routes", source: "Reuters" },
      { text: "Artisanal cobalt mining audit triggers ESG flags for battery supply chains", source: "Financial Times" },
    ],
    summary:
      "DRC produces ~70% of global cobalt supply. Chinese state-backed operators (CMOC, Zijin) control dominant mine positions while armed group activity in eastern provinces creates persistent supply disruption risk. Any escalation in eastern DRC immediately impacts EV battery supply chains globally.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "zambia-copper",
    name: "Zambia — Copper Belt",
    countryIds: ["894"], // Zambia
    signalType: "mixed",
    riskScore: 5.8,
    isPulsing: false,
    mineralTags: ["copper"],
    signals: [
      { text: "Zambia revises mining royalty regime — Copperbelt operators impacted", source: "Reuters" },
      { text: "First Quantum Minerals Kansanshi expansion approved after regulatory delay", source: "Mining Weekly" },
      { text: "Chinese firms increase Zambia copper offtake agreements", source: "Bloomberg" },
    ],
    summary:
      "Zambia is Africa's second-largest copper producer. Regulatory risk elevated by royalty revision negotiations. Chinese firms are deepening offtake relationships, reducing availability for Western supply chains. IMF debt restructuring provides fiscal stability but constrains infrastructure investment.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "zimbabwe-lithium",
    name: "Zimbabwe — Lithium & Rare Earth",
    countryIds: ["716"], // Zimbabwe
    signalType: "risk",
    riskScore: 6.2,
    isPulsing: false,
    mineralTags: ["lithium", "rare_earth"],
    signals: [
      { text: "Zimbabwe lithium export ban forces Chinese processors to establish local operations", source: "Reuters" },
      { text: "Sinomine and Chengxin Lithium expand Arcadia mine stake", source: "Mining.com" },
      { text: "ZANU-PF mineral nationalization signals increase ahead of elections", source: "The Diplomat" },
    ],
    summary:
      "Zimbabwe has significant lithium reserves and has implemented export bans to force value-add processing. Chinese firms have moved quickly to establish in-country refining, effectively locking in supply. Political risk from ZANU-PF resource nationalism creates long-term contract insecurity for non-Chinese operators.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "namibia-minerals",
    name: "Namibia — Green Minerals",
    countryIds: ["516"], // Namibia
    signalType: "opportunity",
    riskScore: 3.2,
    isPulsing: false,
    mineralTags: ["rare_earth"],
    signals: [
      { text: "Namibia Green Hydrogen project advances; EU offtake agreements signed", source: "Reuters" },
      { text: "Lodestone Namibia rare earth project enters feasibility stage", source: "Mining Weekly" },
      { text: "Namibia positions as DRC cobalt alternative for Western supply chains", source: "Financial Times" },
    ],
    summary:
      "Namibia is emerging as a stable, governance-rated alternative source for critical minerals in southern Africa. EU and German government partnerships provide financing and offtake security. Rare earth and green hydrogen projects position Namibia as a strategic non-Chinese supply chain node.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "south-africa-pgm",
    name: "South Africa — PGM Belt",
    countryIds: ["710"], // South Africa
    signalType: "mixed",
    riskScore: 5.2,
    isPulsing: false,
    mineralTags: ["pgm"],
    signals: [
      { text: "Load-shedding at Eskom reduces platinum group metals smelter output", source: "Mining Weekly" },
      { text: "NUM union strike threatens Implats and Anglo Platinum operations", source: "Reuters" },
      { text: "South Africa seeks to renegotiate critical minerals trade terms with US", source: "Bloomberg" },
    ],
    summary:
      "South Africa accounts for ~70% of global platinum group metals production. Persistent electricity supply disruptions from Eskom directly reduce smelter output. Labor relations in the mining sector remain contentious. US–South Africa trade negotiations around critical minerals create both risk and strategic opportunity.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "chile-lithium",
    name: "Chile — Lithium Triangle",
    countryIds: ["152"], // Chile
    signalType: "mixed",
    riskScore: 5.6,
    isPulsing: false,
    mineralTags: ["lithium", "copper"],
    signals: [
      { text: "Boric lithium strategy: Codelco to hold 51% in new joint ventures", source: "Reuters" },
      { text: "SQM–Codelco Atacama partnership terms finalized; Chinese offtake retained", source: "Bloomberg" },
      { text: "Chile copper strike at El Teniente mine enters day 12", source: "Reuters" },
    ],
    summary:
      "Chile holds the world's largest lithium reserves. President Boric's partial nationalization strategy creates uncertainty for foreign operators while maintaining Chinese offtake relationships. Copper strikes at major mines are an ongoing labor risk. The Atacama partnership model will define global lithium market structure through 2030.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "argentina-lithium",
    name: "Argentina — Lithium & Copper",
    countryIds: ["032"], // Argentina
    signalType: "risk",
    riskScore: 6.1,
    isPulsing: false,
    mineralTags: ["lithium", "copper"],
    signals: [
      { text: "Milei RIGI regime attracts $5bn lithium investment commitments", source: "Reuters" },
      { text: "Ganfeng Lithium Cauchari expansion faces provincial royalty dispute", source: "Mining.com" },
      { text: "Argentina peso depreciation increases dollar-cost base for operators", source: "Bloomberg" },
    ],
    summary:
      "Argentina's Puna region hosts significant lithium reserves. President Milei's deregulatory RIGI investment regime has attracted renewed foreign capital but provincial governments are asserting royalty claims. Chinese lithium processors (Ganfeng, CATL) have established dominant positions in the most productive salares.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "bolivia-lithium",
    name: "Bolivia — Uyuni Lithium",
    countryIds: ["068"], // Bolivia
    signalType: "risk",
    riskScore: 6.8,
    isPulsing: false,
    mineralTags: ["lithium"],
    signals: [
      { text: "CATL Uyuni direct lithium extraction pilot reports extraction rate concerns", source: "Mining.com" },
      { text: "Bolivia refuses third-party audit of YLB lithium contracts with Chinese partners", source: "Reuters" },
      { text: "Post-coup political uncertainty delays Uyuni infrastructure investment", source: "The Diplomat" },
    ],
    summary:
      "Bolivia hosts the world's largest estimated lithium reserves at Uyuni but extraction remains below potential due to technical and political challenges. CATL's direct lithium extraction pilot is underperforming. State-controlled YLB has concentrated relationships with Chinese operators, excluding Western access to this strategic deposit.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "peru-copper",
    name: "Peru — Copper & Silver",
    countryIds: ["604"], // Peru
    signalType: "risk",
    riskScore: 6.5,
    isPulsing: false,
    mineralTags: ["copper"],
    signals: [
      { text: "Las Bambas community blockade enters third week; CMOC export halted", source: "Reuters" },
      { text: "Peru declares emergency at Tia Maria; social conflict risk elevated", source: "Mining Weekly" },
      { text: "Zijin Mining increases Southern Copper offtake under new framework", source: "Bloomberg" },
    ],
    summary:
      "Peru is the world's second-largest copper producer. Community opposition to mining projects has blocked production at multiple major mines including Las Bambas (owned by CMOC). Chinese operators control significant offtake from Peruvian copper production. Social license risk is the primary near-term concern for production continuity.",
    lastUpdated: new Date().toISOString(),
  },
  {
    id: "brazil-rare-earth",
    name: "Brazil — Rare Earth & Nickel",
    countryIds: ["076"], // Brazil
    signalType: "opportunity",
    riskScore: 4.1,
    isPulsing: false,
    mineralTags: ["rare_earth", "nickel"],
    signals: [
      { text: "Brazil rare earth reserves second only to China — new processing capacity announced", source: "Reuters" },
      { text: "Lula government formalizes critical minerals partnership with EU", source: "Bloomberg" },
      { text: "Vale nickel operations face environmental license delays in Para state", source: "Mining Weekly" },
    ],
    summary:
      "Brazil has the world's second-largest rare earth reserves and significant nickel production. President Lula is actively positioning Brazil as a non-Chinese critical minerals supplier to the EU and US. Vale's nickel operations are a key strategic asset. Environmental licensing reform will be critical to unlocking Brazil's full minerals potential.",
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
  const debugLogged = useRef(false);

  // Build country-ID → region lookup
  const countryMap = useMemo(
    () =>
      Object.fromEntries(
        regions.flatMap((r) => r.countryIds.map((id) => [id, r]))
      ),
    [regions]
  );

  // ONE-SHOT DIAGNOSTIC — remove after confirming IDs
  useEffect(() => {
    console.log("[AgentSafe debug] countryMap keys:", Object.keys(countryMap).sort());
    console.log("[AgentSafe debug] regions count:", regions.length);
  }, [countryMap, regions.length]);

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
    }

    // Minerals query — DRC cobalt belt and Chile lithium triangle
    try {
      const mRes = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query:
            "Analyse DRC cobalt supply disruption risk, Chile lithium nationalization, and Chinese mining activity across Africa and LatAm",
        }),
      });
      if (mRes.ok) {
        const mBrief: BriefResponse = await mRes.json();
        if (mBrief.status === "complete") {
          const ma = mBrief.risk_assessments?.[0];
          if (ma) {
            setRegions((prev) =>
              prev.map((r) => {
                if (r.id === "drc-cobalt") {
                  return {
                    ...r,
                    riskScore: ma.risk_score ?? r.riskScore,
                    summary: ma.summary ?? r.summary,
                    lastUpdated: new Date().toISOString(),
                    isLive: true,
                    briefId: mBrief.brief_id,
                  };
                }
                if (r.id === "chile-lithium") {
                  return {
                    ...r,
                    riskScore: Math.max(3, (ma.risk_score ?? r.riskScore) - 1),
                    lastUpdated: new Date().toISOString(),
                    isLive: true,
                  };
                }
                return r;
              })
            );
          }
        }
      }
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
                    // ONE-SHOT DIAGNOSTIC — log first geo.id and a few lookups
                    if (!debugLogged.current) {
                      debugLogged.current = true;
                      console.log("[AgentSafe debug] first geo.id:", geo.id, "typeof:", typeof geo.id);
                      console.log("[AgentSafe debug] DRC lookup (key '180'):", countryMap["180"]);
                      console.log("[AgentSafe debug] sample geo ids:", geographies.slice(0,5).map(g => g.id));
                    }
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

              {/* Mineral Exposure */}
              {selected.mineralTags && selected.mineralTags.length > 0 && (
                <div className="px-4 py-4 border-b border-[#1a2236]">
                  <p className="text-[9px] font-mono text-[#5c6882] uppercase tracking-widest mb-2">
                    Mineral Exposure
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {selected.mineralTags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[10px] font-mono px-2 py-0.5 rounded border"
                        style={{
                          color: "#e4a84b",
                          borderColor: "rgba(228,168,75,0.25)",
                          background: "rgba(228,168,75,0.10)",
                        }}
                      >
                        {tag.replace(/_/g, " ")}
                      </span>
                    ))}
                  </div>
                </div>
              )}

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
