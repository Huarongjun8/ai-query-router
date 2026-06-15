"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { UserButton } from "@clerk/nextjs";
import Sidebar from "@/components/Sidebar";
import {
  submitQuery,
  waitForBrief,
  type BriefResponse,
  type MarketSignal,
  type GDELTSignal,
} from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────────────────

type AppState = "idle" | "loading" | "complete" | "error";

// ── Constants ──────────────────────────────────────────────────────────────────

const GRISK_INDICES = [
  { label: "GRISK-CN",     value: 72 },
  { label: "GRISK-TW",     value: 85 },
  { label: "GRISK-SEMI",   value: 68 },
  { label: "GRISK-SUPPLY", value: 61 },
];

const QUICK_QUERIES = [
  "China Taiwan invasion risk",
  "US China semiconductor export controls",
  "PBOC monetary policy signals",
  "BYD supply chain disruption",
  "Hong Kong financial stability",
];

const HORIZON_LABELS: Record<string, string> = {
  immediate:   "Immediate",
  short_term:  "Short-term",
  medium_term: "Medium-term",
  long_term:   "Long-term",
};

// ── Color helpers ──────────────────────────────────────────────────────────────

function griskColor(v: number): string {
  if (v > 70) return "#E24B4A";
  if (v >= 40) return "#E4A84B";
  return "#1D9E75";
}

function scoreColor(s: number): string {
  if (s >= 7) return "#E24B4A";
  if (s >= 4) return "#E4A84B";
  return "#1D9E75";
}

function probabilityColor(p: number): string {
  if (p >= 0.6) return "#E24B4A";
  if (p >= 0.3) return "#E4A84B";
  return "#1D9E75";
}

function formatVolume(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(0)}K`;
  return v > 0 ? String(v) : "—";
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string | number;
  unit?: string;
  color: string;
}) {
  return (
    <div className="bg-surface-high rounded-lg p-3 flex flex-col gap-1 border border-[#1e2a3f]">
      <span className="text-[9px] font-mono text-ink-muted uppercase tracking-widest">
        {label}
      </span>
      <span
        className="text-xl font-bold font-mono leading-tight"
        style={{ color }}
      >
        {value}
        {unit && (
          <span className="text-xs font-normal text-ink-muted ml-0.5">
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

function MarketSignalRow({ signal }: { signal: MarketSignal }) {
  const pct = Math.round(signal.probability * 100);
  const color = probabilityColor(signal.probability);
  return (
    <div className="py-2.5 border-b border-surface-high last:border-0">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <a
          href={signal.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[11px] text-ink-secondary hover:text-accent-teal transition-colors leading-snug flex-1 line-clamp-2"
        >
          {signal.question}
        </a>
        <span
          className="text-sm font-mono font-bold shrink-0"
          style={{ color }}
        >
          {pct}%
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-surface rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <span className="text-[10px] font-mono text-ink-muted shrink-0">
          {formatVolume(signal.volume)}
        </span>
      </div>
    </div>
  );
}

function GDELTPanel({ signal }: { signal: GDELTSignal }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-mono">
          <span className="text-ink-primary font-bold">
            {signal.article_count}
          </span>
          <span className="text-ink-muted"> articles · 48h</span>
        </span>
        {signal.avg_tone !== 0 && (
          <span className="text-xs font-mono text-ink-muted">
            tone{" "}
            <span
              className="font-semibold"
              style={{
                color:
                  signal.avg_tone < -3
                    ? "#E24B4A"
                    : signal.avg_tone < 0
                    ? "#E4A84B"
                    : "#1D9E75",
              }}
            >
              {signal.avg_tone.toFixed(2)}
            </span>
          </span>
        )}
      </div>
      {signal.top_sources.length > 0 && (
        <div className="flex flex-col gap-1">
          {signal.top_sources.slice(0, 4).map((src) => (
            <span
              key={src}
              className="text-[10px] font-mono text-ink-muted px-1.5 py-0.5 rounded bg-surface border border-surface-high truncate"
            >
              {src}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────────

export default function ChinaRiskDashboard() {
  const [query, setQuery]               = useState("");
  const [appState, setAppState]         = useState<AppState>("idle");
  const [brief, setBrief]               = useState<BriefResponse | null>(null);
  const [pollStatus, setPollStatus]     = useState("");
  const [activeQuery, setActiveQuery]   = useState<string | undefined>();
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [mounted, setMounted]           = useState(false);

  const abortRef     = useRef(false);
  const startTimeRef = useRef<number>(0);

  useEffect(() => setMounted(true), []);

  const runQuery = useCallback(async (q: string) => {
    if (!q.trim()) return;
    abortRef.current = false;
    startTimeRef.current = Date.now();
    setAppState("loading");
    setActiveQuery(q);
    setPollStatus("Submitting…");
    setBrief(null);
    setProcessingTime(null);

    try {
      const { brief_id } = await submitQuery(q);
      const result = await waitForBrief(
        brief_id,
        (interim) => {
          if (!abortRef.current)
            setPollStatus(
              interim.status === "processing"
                ? "Agents running — collecting & analysing signals…"
                : interim.status
            );
        },
        2000
      );
      if (!abortRef.current) {
        setProcessingTime(
          Math.round((Date.now() - startTimeRef.current) / 1000)
        );
        setBrief(result);
        setAppState("complete");
      }
    } catch (err) {
      if (!abortRef.current) {
        setPollStatus(err instanceof Error ? err.message : String(err));
        setAppState("error");
      }
    }
  }, []);

  const handleSelect = useCallback(
    (q: string) => {
      setQuery(q);
      runQuery(q);
    },
    [runQuery]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runQuery(query);
  };

  // ── Derived brief data ─────────────────────────────────────────────────────
  const assessment    = brief?.risk_assessments?.[0];
  const riskScore     = assessment?.risk_score   ?? brief?.risk_score   ?? 0;
  const likelihood    = assessment?.likelihood   ?? brief?.likelihood   ?? 0;
  const severity      = assessment?.severity     ?? brief?.severity     ?? 0;
  const timeHorizon   = assessment?.time_horizon ?? brief?.time_horizon ?? "";
  const marketSignals = brief?.market_signals?.length
    ? brief.market_signals
    : (assessment?.market_signals ?? []);
  const gdeltSignal   =
    brief?.gdelt_signal ?? assessment?.gdelt_signal ?? null;
  const complianceKey = brief?.compliance_status ?? "not_checked";

  const complianceCls =
    complianceKey === "passed"
      ? "bg-accent-teal/20 text-accent-teal border-accent-teal/40"
      : complianceKey === "failed"
      ? "bg-accent-red/20 text-accent-red border-accent-red/40"
      : "bg-ink-muted/20 text-ink-secondary border-ink-muted/30";

  const modelTier =
    riskScore >= 7 || (activeQuery ?? "").split(" ").length > 10
      ? "Sonnet"
      : "Haiku";

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="h-screen flex flex-col overflow-hidden">

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 h-14 flex items-center gap-4 px-4 bg-surface-mid border-b border-surface-high z-10">

        {/* Brand */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-7 h-7 rounded bg-accent-teal/20 border border-accent-teal/40 flex items-center justify-center">
            <span className="text-accent-teal text-[10px] font-bold font-mono">
              AS
            </span>
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-[11px] font-bold text-ink-primary font-mono tracking-wider">
              AGENTSAFE
            </span>
            <span className="text-[8px] font-mono text-accent-teal tracking-widest uppercase">
              CHINARISK
            </span>
          </div>
        </div>

        <div className="w-px h-6 bg-surface-high shrink-0" />

        {/* GRISK indices + TWD/USD */}
        <div className="hidden md:flex items-center gap-4 shrink-0">
          {GRISK_INDICES.map(({ label, value }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="text-[9px] font-mono text-ink-muted tracking-widest">
                {label}
              </span>
              <span
                className="text-sm font-mono font-bold tabular-nums"
                style={{ color: griskColor(value) }}
              >
                {value}
              </span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] font-mono text-ink-muted tracking-widest">
              TWD/USD
            </span>
            <span className="text-sm font-mono font-bold text-accent-amber tabular-nums">
              29.84
            </span>
          </div>
        </div>

        <div className="hidden md:block w-px h-6 bg-surface-high shrink-0" />

        {/* Query input */}
        <form
          onSubmit={handleSubmit}
          className="flex-1 flex items-center gap-2 min-w-0"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter China risk query…"
            disabled={appState === "loading"}
            className="flex-1 min-w-0 bg-surface border border-surface-high rounded px-3 py-1.5 text-sm text-ink-primary placeholder:text-ink-muted focus:outline-none focus:border-accent-teal/50 font-mono disabled:opacity-50 transition-colors"
          />
          <button
            type="submit"
            disabled={!query.trim() || appState === "loading"}
            className="shrink-0 px-4 py-1.5 rounded bg-accent-teal text-surface text-sm font-bold font-mono disabled:opacity-40 hover:brightness-110 active:scale-95 transition-all"
          >
            {appState === "loading" ? "…" : "ANALYZE"}
          </button>
        </form>

        {/* User avatar */}
        {mounted && (
          <div className="shrink-0">
            <UserButton
              appearance={{ elements: { avatarBox: "w-6 h-6" } }}
            />
          </div>
        )}
      </header>

      {/* ── 3-column body ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden min-h-0">

        {/* Left panel — Active Signals (200px) */}
        <div className="w-[200px] shrink-0 border-r border-surface-high hidden md:block">
          <Sidebar onSelectQuery={handleSelect} activeQuery={activeQuery} />
        </div>

        {/* Center panel (flex-1) */}
        <main className="flex-1 overflow-y-auto min-w-0 px-5 py-4">

          {/* ── Idle ──────────────────────────────────────────────────────── */}
          {appState === "idle" && (
            <div className="flex flex-col items-center justify-center min-h-full gap-6 text-center py-12">
              <div>
                <p className="text-ink-primary text-lg font-semibold mb-2">
                  China Risk Intelligence for Institutional Investors
                </p>
                <p className="text-ink-muted text-sm max-w-md">
                  Mandarin + English source analysis across prediction markets,
                  news signals, and Chinese-language feeds.
                </p>
              </div>
              <div className="flex flex-col gap-2 w-full max-w-lg">
                <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest text-left">
                  Quick queries
                </p>
                {QUICK_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setQuery(q);
                      runQuery(q);
                    }}
                    className="text-left text-sm text-ink-secondary border border-surface-high rounded-lg px-4 py-2.5 hover:border-accent-teal/40 hover:text-ink-primary hover:bg-surface-mid transition-all font-mono"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ── Loading ───────────────────────────────────────────────────── */}
          {appState === "loading" && (
            <div className="flex flex-col items-center justify-center min-h-full gap-4 py-12">
              <div className="relative w-10 h-10">
                <div className="absolute inset-0 rounded-full border-2 border-accent-teal/20" />
                <div className="absolute inset-0 rounded-full border-t-2 border-accent-teal animate-spin" />
              </div>
              <div className="flex flex-col items-center gap-1">
                <p className="text-sm text-ink-secondary font-mono">
                  {pollStatus}
                </p>
                <p className="text-xs text-ink-muted font-mono">
                  collect → translate → risk → compliance → report
                </p>
              </div>
            </div>
          )}

          {/* ── Error ─────────────────────────────────────────────────────── */}
          {appState === "error" && (
            <div className="flex flex-col items-center justify-center min-h-full py-12">
              <div className="rounded-lg border border-accent-red/30 bg-accent-red/5 p-5 max-w-md w-full">
                <p className="text-sm font-semibold text-accent-red mb-1">
                  Pipeline Error
                </p>
                <p className="text-xs text-ink-muted font-mono break-all">
                  {pollStatus}
                </p>
                <button
                  onClick={() => activeQuery && runQuery(activeQuery)}
                  className="mt-3 text-xs text-accent-teal hover:underline font-mono"
                >
                  Retry
                </button>
              </div>
            </div>
          )}

          {/* ── Complete ──────────────────────────────────────────────────── */}
          {appState === "complete" && brief && (
            <div className="animate-fade-up flex flex-col gap-4 max-w-2xl">

              {/* Title + meta + compliance badge */}
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div className="flex-1 min-w-0">
                  <h1 className="text-base font-semibold text-ink-primary leading-snug">
                    {brief.title ?? "Intelligence Brief"}
                  </h1>
                  <div className="flex items-center gap-3 mt-1 flex-wrap">
                    {brief.generated_at && (
                      <span className="text-[10px] font-mono text-ink-muted">
                        {new Date(brief.generated_at).toLocaleString()}
                      </span>
                    )}
                    {processingTime != null && (
                      <span className="text-[10px] font-mono text-ink-muted">
                        {processingTime}s
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-ink-muted">
                      {modelTier}
                    </span>
                  </div>
                </div>
                <span
                  className={`shrink-0 text-[10px] font-mono px-2 py-0.5 rounded uppercase tracking-wide border ${complianceCls}`}
                >
                  {complianceKey.replace(/_/g, " ")}
                </span>
              </div>

              {/* 4 metric cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <MetricCard
                  label="Threat"
                  value={severity}
                  unit="/10"
                  color={scoreColor(severity)}
                />
                <MetricCard
                  label="Risk Score"
                  value={riskScore}
                  unit="/10"
                  color={scoreColor(riskScore)}
                />
                <MetricCard
                  label="Likelihood"
                  value={likelihood}
                  unit="/10"
                  color={scoreColor(likelihood)}
                />
                <MetricCard
                  label="Horizon"
                  value={HORIZON_LABELS[timeHorizon] ?? "—"}
                  color="#9BA8C0"
                />
              </div>

              {/* Situation Assessment */}
              {(brief.executive_summary ?? assessment?.summary) && (
                <div className="bg-surface-mid border border-surface-high rounded-lg p-4">
                  <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest mb-2">
                    Situation Assessment
                  </p>
                  <p className="text-sm text-ink-secondary leading-relaxed whitespace-pre-line">
                    {brief.executive_summary ?? assessment?.summary}
                  </p>
                </div>
              )}

              {/* Recommended Actions */}
              {brief.recommendations && brief.recommendations.length > 0 && (
                <div className="bg-surface-mid border border-surface-high rounded-lg p-4">
                  <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest mb-2">
                    Recommended Actions
                  </p>
                  <ul className="flex flex-col gap-2">
                    {brief.recommendations.map((rec, i) => (
                      <li
                        key={i}
                        className="flex items-start gap-2 text-sm text-ink-secondary"
                      >
                        <span className="text-accent-teal mt-0.5 shrink-0 font-mono">
                          →
                        </span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Affected Sectors */}
              {brief.affected_sectors && brief.affected_sectors.length > 0 && (
                <div className="bg-surface-mid border border-surface-high rounded-lg p-4">
                  <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest mb-2">
                    Affected Sectors
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {brief.affected_sectors.map((s) => (
                      <span
                        key={s}
                        className="text-xs font-mono px-2 py-0.5 rounded bg-accent-teal/10 text-accent-teal border border-accent-teal/25 capitalize"
                      >
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Sources — collapsible */}
              {brief.sources && brief.sources.length > 0 && (
                <details className="bg-surface-mid border border-surface-high rounded-lg overflow-hidden group">
                  <summary className="px-4 py-3 text-[10px] font-mono text-ink-muted uppercase tracking-widest cursor-pointer hover:text-ink-secondary transition-colors list-none flex items-center justify-between">
                    <span>Sources ({brief.sources.length})</span>
                    <span className="text-[8px] group-open:rotate-180 transition-transform">
                      ▼
                    </span>
                  </summary>
                  <div className="px-4 pb-3 flex flex-col gap-1 border-t border-surface-high">
                    {brief.sources.map((src, i) => (
                      <a
                        key={i}
                        href={src}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-mono text-ink-muted hover:text-accent-teal truncate transition-colors py-0.5"
                      >
                        {src}
                      </a>
                    ))}
                  </div>
                </details>
              )}

              {/* Disclaimers */}
              {brief.disclaimers && brief.disclaimers.length > 0 && (
                <div className="rounded-lg border border-accent-amber/20 bg-accent-amber/5 p-3">
                  <p className="text-[9px] font-mono text-accent-amber uppercase tracking-widest mb-1">
                    Disclaimers
                  </p>
                  {brief.disclaimers.map((d, i) => (
                    <p key={i} className="text-xs text-ink-muted">
                      {d}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>

        {/* Right panel — Market Signals + GDELT (220px) */}
        <aside className="w-[220px] shrink-0 border-l border-surface-high overflow-y-auto bg-surface hidden lg:flex flex-col">

          {/* Market Signals */}
          <div className="px-3 py-2.5 border-b border-surface-high shrink-0">
            <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest">
              Market Signals
            </p>
          </div>
          <div className="px-3 py-1 flex-1 min-h-0 overflow-y-auto">
            {marketSignals.length > 0 ? (
              marketSignals.map((sig, i) => (
                <MarketSignalRow key={i} signal={sig} />
              ))
            ) : (
              <p className="text-[10px] font-mono text-ink-muted py-4 text-center leading-relaxed">
                {appState === "idle"
                  ? "Run a query to load signals"
                  : appState === "loading"
                  ? "Loading…"
                  : "No market signals returned"}
              </p>
            )}
          </div>

          {/* GDELT Signal */}
          <div className="border-t border-surface-high shrink-0">
            <div className="px-3 py-2.5 border-b border-surface-high">
              <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest">
                GDELT Signal
              </p>
            </div>
            <div className="px-3 py-3">
              {gdeltSignal ? (
                <GDELTPanel signal={gdeltSignal} />
              ) : (
                <p className="text-[10px] font-mono text-ink-muted leading-relaxed">
                  {appState === "idle"
                    ? "Run a query to load"
                    : appState === "loading"
                    ? "Loading…"
                    : "No GDELT data"}
                </p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
