"use client";

import { useState, useRef, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import BriefDisplay from "@/components/BriefDisplay";
import MineralsFilter, {
  type FilterState,
  EMPTY_FILTERS,
  activeFilterCount,
} from "@/components/MineralsFilter";
import { submitQuery, waitForBrief, deriveTags, type BriefResponse } from "@/lib/api";

type AppState = "idle" | "loading" | "complete" | "error";

const POLL_STATUS_LABELS: Record<string, string> = {
  pending: "Queued — pipeline starting…",
  processing: "Agents running — collecting & analysing signals…",
  complete: "Complete",
  error: "Error",
};

const SEVERITY_BADGE: Record<string, string> = {
  high:   "bg-accent-red/10 text-accent-red border-accent-red/25",
  medium: "bg-accent-amber/10 text-accent-amber border-accent-amber/25",
  low:    "bg-accent-teal/10 text-accent-teal border-accent-teal/25",
};

function briefMatchesFilters(brief: BriefResponse, filters: FilterState): boolean {
  const tags = deriveTags(brief);
  if (filters.mineral.length > 0 && !filters.mineral.some((m) => tags.mineral.includes(m))) return false;
  if (filters.region.length > 0 && !filters.region.some((r) => tags.region.includes(r))) return false;
  if (filters.risk_type.length > 0 && !filters.risk_type.some((t) => tags.risk_type.includes(t))) return false;
  if (filters.severity && tags.severity !== filters.severity) return false;
  return true;
}

function BriefCard({
  brief,
  isSelected,
  onClick,
}: {
  brief: BriefResponse;
  isSelected: boolean;
  onClick: () => void;
}) {
  const tags = deriveTags(brief);
  const score =
    brief.risk_assessments?.[0]?.risk_score ?? brief.risk_score ?? 0;
  const scoreColor =
    score >= 7 ? "#E24B4A" : score >= 4 ? "#E4A84B" : "#1D9E75";

  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-xl border px-4 py-3 transition-all ${
        isSelected
          ? "border-accent-teal/50 bg-surface-high"
          : "border-surface-high bg-surface-mid hover:border-ink-muted/40 hover:bg-surface-high"
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Score */}
        <span
          className="shrink-0 text-lg font-bold font-mono tabular-nums leading-tight mt-0.5"
          style={{ color: scoreColor }}
        >
          {score}
          <span className="text-xs font-normal text-ink-muted">/10</span>
        </span>

        {/* Title + tags */}
        <div className="flex-1 min-w-0 flex flex-col gap-1.5">
          <p className="text-sm font-medium text-ink-primary leading-snug line-clamp-1">
            {brief.title ?? "Intelligence Brief"}
          </p>

          {/* Tag badges */}
          <div className="flex flex-wrap gap-1">
            {tags.mineral.map((m) => (
              <span
                key={m}
                className="text-[10px] px-1.5 py-0.5 rounded font-mono border bg-accent-amber/10 text-accent-amber border-accent-amber/25"
              >
                {m.replace(/_/g, " ")}
              </span>
            ))}
            {tags.region.map((r) => (
              <span
                key={r}
                className="text-[10px] px-1.5 py-0.5 rounded font-mono border bg-surface border-surface-high text-ink-secondary"
              >
                {r}
              </span>
            ))}
            {tags.risk_type.map((t) => (
              <span
                key={t}
                className="text-[10px] px-1.5 py-0.5 rounded font-mono border bg-accent-red/10 text-accent-red border-accent-red/20"
              >
                {t.replace(/_/g, " ")}
              </span>
            ))}
            {tags.severity && (
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-mono border ${
                  SEVERITY_BADGE[tags.severity] ?? SEVERITY_BADGE.low
                }`}
              >
                {tags.severity}
              </span>
            )}
          </div>

          {brief.generated_at && (
            <p className="text-[10px] font-mono text-ink-muted">
              {new Date(brief.generated_at).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </p>
          )}
        </div>
      </div>
    </button>
  );
}

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<AppState>("idle");
  const [pollStatus, setPollStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [activeQuery, setActiveQuery] = useState<string | undefined>();

  // Brief history
  const [briefs, setBriefs] = useState<BriefResponse[]>([]);
  const [selectedBriefId, setSelectedBriefId] = useState<string | null>(null);

  // Filters
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [filterOpen, setFilterOpen] = useState(false);

  const abortRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const runQuery = useCallback(async (q: string) => {
    if (!q.trim()) return;

    abortRef.current = false;
    setState("loading");
    setError(null);
    setPollStatus("Submitting query…");
    setActiveQuery(q);

    try {
      const { brief_id } = await submitQuery(q);
      setPollStatus("Submitted — waiting for pipeline…");

      const result = await waitForBrief(
        brief_id,
        (interim) => {
          if (abortRef.current) return;
          setPollStatus(
            POLL_STATUS_LABELS[interim.status] ?? `Status: ${interim.status}`
          );
        },
        2000
      );

      if (!abortRef.current) {
        setBriefs((prev) => [
          result,
          ...prev.filter((b) => b.brief_id !== result.brief_id),
        ]);
        setSelectedBriefId(result.brief_id);
        setState("complete");
      }
    } catch (err: unknown) {
      if (!abortRef.current) {
        setError(err instanceof Error ? err.message : String(err));
        setState("error");
      }
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runQuery(query);
  };

  const handleSidebarSelect = (q: string) => {
    setQuery(q);
    runQuery(q);
    textareaRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runQuery(query);
    }
  };

  const selectedBrief = briefs.find((b) => b.brief_id === selectedBriefId) ?? null;
  const filteredBriefs = briefs.filter((b) => briefMatchesFilters(b, filters));
  const filterCount = activeFilterCount(filters);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="border-b border-surface-high px-6 py-3 flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded bg-accent-teal/20 border border-accent-teal/40 flex items-center justify-center">
            <span className="text-accent-teal text-[10px] font-bold font-mono">AS</span>
          </div>
          <span className="text-sm font-semibold text-ink-primary tracking-tight">
            AgentSafe
          </span>
          <span className="text-xs font-mono text-ink-muted border border-surface-high rounded px-1.5 py-0.5">
            BETA
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-ink-muted font-mono hidden sm:inline">
            US-China Risk Intelligence
          </span>
          <div
            className={`w-2 h-2 rounded-full ${
              state === "loading"
                ? "bg-accent-amber pulse-glow"
                : state === "complete"
                ? "bg-accent-teal"
                : state === "error"
                ? "bg-accent-red"
                : "bg-ink-muted"
            }`}
          />
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="hidden md:flex w-64 shrink-0 border-r border-surface-high px-3 py-4 flex-col">
          <Sidebar onSelectQuery={handleSidebarSelect} activeQuery={activeQuery} />
        </div>

        {/* Content */}
        <main className="flex-1 overflow-y-auto px-4 sm:px-8 py-6 flex flex-col gap-6 max-w-4xl mx-auto w-full">
          {/* Query input */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-mono text-ink-muted uppercase tracking-widest">
              Intelligence Query
            </label>
            <form onSubmit={handleSubmit} className="flex flex-col gap-2">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. Analyse escalation risk from latest US semiconductor export controls on China…"
                rows={3}
                disabled={state === "loading"}
                className="
                  w-full bg-surface-mid border border-surface-high rounded-xl px-4 py-3
                  text-sm text-ink-primary placeholder:text-ink-muted
                  focus:outline-none focus:border-accent-teal/60 focus:ring-1 focus:ring-accent-teal/30
                  resize-none transition-colors disabled:opacity-50
                "
              />
              <div className="flex items-center justify-between">
                <span className="text-xs text-ink-muted">
                  ↵ Enter to submit · Shift+Enter for new line
                </span>
                <button
                  type="submit"
                  disabled={!query.trim() || state === "loading"}
                  className="
                    px-5 py-2 rounded-lg text-sm font-medium transition-all
                    bg-accent-teal text-surface disabled:opacity-40
                    hover:brightness-110 active:scale-95
                    disabled:cursor-not-allowed
                  "
                >
                  {state === "loading" ? "Analysing…" : "Analyse"}
                </button>
              </div>
            </form>
          </div>

          {/* Filter + brief history — shown once at least one brief exists */}
          {briefs.length > 0 && (
            <div className="flex flex-col gap-3">
              {/* Filter toolbar */}
              <MineralsFilter
                filters={filters}
                onChange={setFilters}
                isOpen={filterOpen}
                onToggleOpen={() => setFilterOpen((o) => !o)}
              />

              {/* Brief history separator */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-ink-muted uppercase tracking-widest">
                  Brief history
                </span>
                <span className="text-[10px] font-mono text-ink-muted">
                  {filteredBriefs.length}/{briefs.length}
                  {filterCount > 0 && " filtered"}
                </span>
              </div>

              {/* Cards */}
              {filteredBriefs.length > 0 ? (
                <div className="flex flex-col gap-2">
                  {filteredBriefs.map((b) => (
                    <BriefCard
                      key={b.brief_id}
                      brief={b}
                      isSelected={b.brief_id === selectedBriefId}
                      onClick={() => setSelectedBriefId(b.brief_id)}
                    />
                  ))}
                </div>
              ) : (
                <p className="text-xs text-ink-muted text-center py-4 font-mono">
                  No briefs match the active filters.
                </p>
              )}
            </div>
          )}

          {/* Loading state */}
          {state === "loading" && (
            <div
              className={`flex flex-col items-center gap-4 py-12 ${
                briefs.length > 0 ? "" : "animate-fade-up"
              }`}
            >
              <div className="relative w-12 h-12">
                <div className="absolute inset-0 rounded-full border-2 border-accent-teal/20" />
                <div className="absolute inset-0 rounded-full border-t-2 border-accent-teal animate-spin" />
              </div>
              <div className="flex flex-col items-center gap-1">
                <p className="text-sm text-ink-secondary font-medium">{pollStatus}</p>
                <p className="text-xs text-ink-muted font-mono">
                  Polling every 2s · agents: collect → translate → risk → compliance → report
                </p>
              </div>
            </div>
          )}

          {/* Error state */}
          {state === "error" && error && (
            <div className="animate-fade-up rounded-xl border border-accent-red/30 bg-accent-red/5 p-5">
              <p className="text-sm font-semibold text-accent-red mb-1">Pipeline Error</p>
              <p className="text-xs text-ink-muted font-mono break-all">{error}</p>
              <button
                onClick={() => runQuery(query)}
                className="mt-3 text-xs text-accent-teal hover:underline"
              >
                Retry
              </button>
            </div>
          )}

          {/* Idle state — only when no briefs yet */}
          {state === "idle" && briefs.length === 0 && (
            <div className="flex flex-col items-center gap-3 py-16 text-center">
              <p className="text-ink-muted text-sm max-w-md">
                Submit a query or select a watchlist from the sidebar to generate a
                geopolitical risk intelligence brief.
              </p>
              <p className="text-xs font-mono text-ink-muted/60">
                Powered by multi-agent analysis · English + Mandarin sources
              </p>
            </div>
          )}

          {/* Full brief detail — shown for the selected card */}
          {selectedBrief && state !== "loading" && (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 border-t border-surface-high pt-4">
                <span className="text-[10px] font-mono text-ink-muted uppercase tracking-widest flex-1">
                  Brief detail
                </span>
                {briefs.length > 1 && (
                  <span className="text-[10px] font-mono text-ink-muted">
                    {briefs.findIndex((b) => b.brief_id === selectedBriefId) + 1} of {briefs.length}
                  </span>
                )}
              </div>
              <BriefDisplay brief={selectedBrief} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
