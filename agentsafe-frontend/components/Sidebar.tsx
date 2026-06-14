"use client";

import { useState, useEffect } from "react";
import { UserButton } from "@clerk/nextjs";

interface WatchlistItem {
  id: string;
  label: string;
  query: string;
  tag: "high" | "medium" | "low";
}

const WATCHLISTS: WatchlistItem[] = [
  {
    id: "taiwan-strait-tension",
    label: "Taiwan strait tension",
    query:
      "Taiwan strait escalation risk and implications for regional financial markets",
    tag: "high",
  },
  {
    id: "us-china-tariffs",
    label: "US-China tariffs",
    query:
      "US-China tariff escalation risks and impact on trade-exposed financial assets",
    tag: "high",
  },
  {
    id: "pboc-policy",
    label: "PBOC policy signals",
    query:
      "PBOC monetary policy signals and People's Bank of China rate decisions affecting global markets",
    tag: "medium",
  },
  {
    id: "china-supply-chain",
    label: "China supply chain",
    query:
      "China supply chain disruption risks for semiconductors, rare earths, and critical materials",
    tag: "medium",
  },
  {
    id: "hong-kong-markets",
    label: "Hong Kong markets",
    query:
      "Hong Kong financial market stability risks and regulatory environment under Beijing's influence",
    tag: "medium",
  },
  {
    id: "xi-policy-signals",
    label: "Xi policy signals",
    query:
      "Xi Jinping policy signals and political direction affecting China's economic and regulatory environment",
    tag: "high",
  },
];

const TAG_STYLES = {
  high: "bg-accent-red/20 text-accent-red border border-accent-red/30",
  medium: "bg-accent-amber/20 text-accent-amber border border-accent-amber/30",
  low: "bg-accent-teal/20 text-accent-teal border border-accent-teal/30",
};

interface SidebarProps {
  onSelectQuery: (query: string) => void;
  activeQuery?: string;
}

export default function Sidebar({ onSelectQuery, activeQuery }: SidebarProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <aside className="w-64 shrink-0 flex flex-col gap-1 pt-2">
      {/* Logo */}
      <div className="flex items-center gap-2 px-1 mb-5">
        <div className="w-6 h-6 rounded bg-accent-teal/20 border border-accent-teal/40 flex items-center justify-center shrink-0">
          <span className="text-accent-teal text-[10px] font-bold font-mono">AS</span>
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-ink-primary tracking-tight">AgentSafe</span>
          <span className="text-[10px] font-mono text-ink-muted tracking-widest uppercase">ChinaRisk</span>
        </div>
      </div>

      <p className="text-xs font-mono text-ink-muted uppercase tracking-widest mb-3 px-1">
        Watchlists
      </p>

      {WATCHLISTS.map((item) => {
        const isActive = activeQuery === item.query;
        return (
          <button
            key={item.id}
            onClick={() => onSelectQuery(item.query)}
            className={`
              w-full text-left px-3 py-2.5 rounded-lg transition-all duration-150
              flex items-center justify-between gap-2 group
              ${
                isActive
                  ? "bg-surface-high border border-accent-teal/40 text-ink-primary"
                  : "text-ink-secondary hover:bg-surface-high hover:text-ink-primary border border-transparent"
              }
            `}
          >
            <span className="text-sm font-medium truncate">{item.label}</span>
            <span
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-wide shrink-0 ${TAG_STYLES[item.tag]}`}
            >
              {item.tag}
            </span>
          </button>
        );
      })}

      <div className="mt-auto pt-6 px-1">
        <div className="border-t border-surface-high pt-4 flex flex-col gap-3">
          <p className="text-[11px] text-ink-muted leading-relaxed">
            Signals monitored across{" "}
            <span className="text-ink-secondary">English + Mandarin</span>{" "}
            sources. Updated on query.
          </p>
          {mounted && (
            <UserButton
              appearance={{
                elements: {
                  avatarBox: "w-7 h-7",
                },
              }}
            />
          )}
        </div>
      </div>
    </aside>
  );
}
