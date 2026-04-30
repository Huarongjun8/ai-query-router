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
    id: "china-exposure",
    label: "China exposure",
    query:
      "Analyse current China market exposure risks for US-listed financial institutions",
    tag: "high",
  },
  {
    id: "us-sanctions",
    label: "US sanctions",
    query:
      "Latest US sanctions developments affecting China technology and financial sectors",
    tag: "high",
  },
  {
    id: "supply-chain",
    label: "Supply chain",
    query:
      "Semiconductor and critical materials supply chain disruption risk from US-China tensions",
    tag: "medium",
  },
  {
    id: "taiwan-strait",
    label: "Taiwan strait",
    query:
      "Taiwan strait escalation risk and implications for regional financial markets",
    tag: "high",
  },
  {
    id: "iran-hormuz",
    label: "Iran / Hormuz",
    query:
      "Iran Strait of Hormuz energy supply disruption risk and oil market implications",
    tag: "medium",
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
