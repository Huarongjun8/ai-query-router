"use client";

import { useState } from "react";
import type { BriefResponse, MarketSignal, GDELTSignal } from "@/lib/api";
import ScoreBar from "./ScoreBar";

/** Renders a string with **bold** markdown as <strong> inline elements. */
function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i} className="text-ink-primary font-semibold">
            {part.slice(2, -2)}
          </strong>
        ) : (
          part
        )
      )}
    </>
  );
}

interface BriefDisplayProps {
  brief: BriefResponse;
}

const HORIZON_LABELS: Record<string, string> = {
  immediate: "Immediate",
  short_term: "Short-term",
  medium_term: "Medium-term",
  long_term: "Long-term",
};

const COMPLIANCE_STYLES: Record<string, string> = {
  passed: "bg-accent-teal/20 text-accent-teal border border-accent-teal/40",
  failed: "bg-accent-red/20 text-accent-red border border-accent-red/40",
  not_checked:
    "bg-ink-muted/20 text-ink-secondary border border-ink-muted/30",
};

function MetricCard({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: number | string;
  unit?: string;
  color: string;
}) {
  return (
    <div className="bg-surface-high rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-ink-muted uppercase tracking-wide font-mono">
        {label}
      </span>
      <span className="text-3xl font-bold" style={{ color }}>
        {value}
        {unit && (
          <span className="text-base font-normal text-ink-muted ml-0.5">
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

function scoreColor(score: number): string {
  if (score <= 4) return "#1D9E75";
  if (score <= 7) return "#E4A84B";
  return "#E24B4A";
}

function probabilityColor(p: number): string {
  if (p < 0.30) return "#1D9E75";   // green — low probability
  if (p < 0.60) return "#E4A84B";   // amber — moderate
  return "#E24B4A";                  // red — high probability
}

function formatVolume(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`;
  return v > 0 ? `${v.toFixed(0)}` : "—";
}

function MarketSignalRow({ signal }: { signal: MarketSignal }) {
  const pct = Math.round(signal.probability * 100);
  const color = probabilityColor(signal.probability);
  return (
    <div className="flex flex-col gap-1.5 py-3 border-b border-surface last:border-0">
      <div className="flex items-start justify-between gap-3">
        <a
          href={signal.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-ink-secondary hover:text-accent-teal transition-colors leading-snug flex-1"
        >
          {signal.question}
        </a>
        <span
          className="shrink-0 text-sm font-mono font-semibold tabular-nums"
          style={{ color }}
        >
          {pct}%
        </span>
      </div>
      <div className="flex items-center gap-3">
        {/* Probability bar */}
        <div className="flex-1 h-1.5 bg-surface rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <span className="text-xs font-mono text-ink-muted shrink-0">
          {formatVolume(signal.volume)}
        </span>
      </div>
    </div>
  );
}

function GDELTPanel({ signal }: { signal: GDELTSignal }) {
  const topSources = signal.top_sources.slice(0, 3);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-xs font-mono text-ink-muted">
          <span className="text-ink-secondary font-semibold">{signal.article_count}</span>
          {" articles · "}
          <span className="text-ink-secondary">48h window</span>
        </span>
        {signal.avg_tone !== 0 && (
          <span className="text-xs font-mono text-ink-muted">
            avg tone{" "}
            <span
              className="font-semibold"
              style={{ color: signal.avg_tone < -3 ? "#E24B4A" : signal.avg_tone < 0 ? "#E4A84B" : "#1D9E75" }}
            >
              {signal.avg_tone.toFixed(2)}
            </span>
          </span>
        )}
      </div>
      {topSources.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {topSources.map((src) => (
            <span
              key={src}
              className="text-xs font-mono px-2 py-0.5 rounded bg-surface border border-surface-high text-ink-muted"
            >
              {src}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function BriefDisplay({ brief }: BriefDisplayProps) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // Metrics live in risk_assessments[0]; fall back to top-level for older API responses
  const assessment = brief.risk_assessments?.[0];
  const riskScore = assessment?.risk_score ?? brief.risk_score ?? 0;
  const likelihood = assessment?.likelihood ?? brief.likelihood ?? 0;
  const severity = assessment?.severity ?? brief.severity ?? 0;
  const reversibility = assessment?.reversibility ?? brief.reversibility ?? 0;
  const timeHorizon = assessment?.time_horizon ?? brief.time_horizon ?? "";

  const complianceKey = brief.compliance_status ?? "not_checked";
  const complianceStyle =
    COMPLIANCE_STYLES[complianceKey] ?? COMPLIANCE_STYLES["not_checked"];

  return (
    <div className="animate-fade-up flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold text-ink-primary leading-snug">
            {brief.title ?? "Intelligence Brief"}
          </h2>
          {brief.generated_at && (
            <p className="text-xs font-mono text-ink-muted">
              {new Date(brief.generated_at).toLocaleString()}
            </p>
          )}
        </div>
        <span
          className={`shrink-0 text-xs font-mono px-2.5 py-1 rounded-full uppercase tracking-wide ${complianceStyle}`}
        >
          {complianceKey.replace("_", " ")}
        </span>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
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
          label="Severity"
          value={severity}
          unit="/10"
          color={scoreColor(severity)}
        />
        <MetricCard
          label="Time Horizon"
          value={HORIZON_LABELS[timeHorizon] ?? "—"}
          color="#9BA8C0"
        />
      </div>

      {/* Score bars */}
      <div className="bg-surface-high rounded-xl p-4 flex flex-col gap-4">
        <ScoreBar label="Risk Score" value={riskScore} color="auto" />
        <ScoreBar label="Likelihood" value={likelihood} color="auto" />
        <ScoreBar label="Severity" value={severity} color="auto" />
        <ScoreBar
          label="Reversibility"
          value={reversibility}
          color="teal"
        />
      </div>

      {/* Market signals */}
      {(() => {
        const signals = brief.market_signals?.length
          ? brief.market_signals
          : (assessment?.market_signals ?? []);
        return signals.length > 0 ? (
          <div className="bg-surface-high rounded-xl p-4">
            <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-1">
              Market Signals ({signals.length})
            </p>
            <div className="divide-y divide-surface">
              {signals.map((sig, i) => (
                <MarketSignalRow key={i} signal={sig} />
              ))}
            </div>
          </div>
        ) : null;
      })()}

      {/* GDELT signal */}
      {(() => {
        const gdelt = brief.gdelt_signal ?? assessment?.gdelt_signal ?? null;
        return gdelt ? (
          <div className="bg-surface-high rounded-xl p-4">
            <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-2">
              News Intelligence · GDELT
            </p>
            <GDELTPanel signal={gdelt} />
          </div>
        ) : null;
      })()}

      {/* Executive summary */}
      {brief.executive_summary && (
        <div className="bg-surface-high rounded-xl p-5">
          <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-3">
            Executive Summary
          </p>
          <p className="text-sm text-ink-secondary leading-relaxed whitespace-pre-line">
            <InlineMarkdown text={brief.executive_summary} />
          </p>
        </div>
      )}

      {/* Sectors + Geo */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {brief.affected_sectors && brief.affected_sectors.length > 0 && (
          <div className="bg-surface-high rounded-xl p-4">
            <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-3">
              Affected Sectors
            </p>
            <div className="flex flex-wrap gap-2">
              {brief.affected_sectors.map((s) => (
                <span
                  key={s}
                  className="text-xs px-2.5 py-1 rounded-full bg-accent-teal/15 text-accent-teal border border-accent-teal/30 font-medium capitalize"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}

        {brief.geographic_scope && brief.geographic_scope.length > 0 && (
          <div className="bg-surface-high rounded-xl p-4">
            <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-3">
              Geographic Scope
            </p>
            <div className="flex flex-wrap gap-2">
              {brief.geographic_scope.map((g) => (
                <span
                  key={g}
                  className="text-xs px-2.5 py-1 rounded-full bg-surface border border-surface-high text-ink-secondary font-medium"
                >
                  {g}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Recommendations */}
      {brief.recommendations && brief.recommendations.length > 0 && (
        <div className="bg-surface-high rounded-xl p-4">
          <p className="text-xs text-ink-muted uppercase tracking-wide font-mono mb-3">
            Recommendations
          </p>
          <ul className="flex flex-col gap-2">
            {brief.recommendations.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-ink-secondary">
                <span className="text-accent-teal mt-0.5 shrink-0">→</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Disclaimers */}
      {brief.disclaimers && brief.disclaimers.length > 0 && (
        <div className="rounded-xl border border-accent-amber/20 bg-accent-amber/5 p-4">
          <p className="text-xs text-accent-amber uppercase tracking-wide font-mono mb-2">
            Disclaimers
          </p>
          <ul className="flex flex-col gap-1">
            {brief.disclaimers.map((d, i) => (
              <li key={i} className="text-xs text-ink-muted">
                {d}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Sources — collapsible */}
      {brief.sources && brief.sources.length > 0 && (
        <div className="bg-surface-high rounded-xl overflow-hidden">
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-surface transition-colors"
          >
            <span className="text-xs text-ink-muted uppercase tracking-wide font-mono">
              Sources ({brief.sources.length})
            </span>
            <span className="text-ink-muted text-sm">
              {sourcesOpen ? "▲" : "▼"}
            </span>
          </button>

          {sourcesOpen && (
            <div className="px-4 pb-4 flex flex-col gap-1 border-t border-surface">
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
          )}
        </div>
      )}
    </div>
  );
}
