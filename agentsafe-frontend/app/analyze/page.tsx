"use client";

import { useState, useRef, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import BriefDisplay from "@/components/BriefDisplay";
import { submitQuery, waitForBrief, type BriefResponse } from "@/lib/api";

type AppState = "idle" | "loading" | "complete" | "error";

const POLL_STATUS_LABELS: Record<string, string> = {
  pending: "Queued — pipeline starting…",
  processing: "Agents running — collecting & analysing signals…",
  complete: "Complete",
  error: "Error",
};

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [state, setState] = useState<AppState>("idle");
  const [pollStatus, setPollStatus] = useState("");
  const [brief, setBrief] = useState<BriefResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeQuery, setActiveQuery] = useState<string | undefined>();
  const abortRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const runQuery = useCallback(async (q: string) => {
    if (!q.trim()) return;

    abortRef.current = false;
    setState("loading");
    setError(null);
    setBrief(null);
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
        setBrief(result);
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

          {/* Loading state */}
          {state === "loading" && (
            <div className="animate-fade-up flex flex-col items-center gap-4 py-12">
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

          {/* Brief output */}
          {state === "complete" && brief && (
            <BriefDisplay brief={brief} />
          )}

          {/* Idle state */}
          {state === "idle" && (
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
        </main>
      </div>
    </div>
  );
}
