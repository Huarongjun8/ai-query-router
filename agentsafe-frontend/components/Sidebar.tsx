"use client";

interface Signal {
  id: string;
  label: string;
  subtitle: string;
  score: number;
  query: string;
}

const SIGNALS: Signal[] = [
  {
    id: "taiwan-strait",
    label: "Taiwan Strait",
    subtitle: "Military escalation",
    score: 9.0,
    query:
      "Taiwan strait escalation risk and implications for regional financial markets",
  },
  {
    id: "us-china-tech",
    label: "US-China Tech",
    subtitle: "Semiconductor decoupling",
    score: 8.0,
    query:
      "US-China semiconductor export controls and technology decoupling risks",
  },
  {
    id: "trade-war",
    label: "Trade War",
    subtitle: "Tariff escalation",
    score: 7.5,
    query:
      "US-China tariff escalation risks and impact on trade-exposed financial assets",
  },
  {
    id: "pboc-policy",
    label: "PBOC Policy",
    subtitle: "Monetary signals",
    score: 6.5,
    query:
      "PBOC monetary policy signals and People's Bank of China rate decisions affecting global markets",
  },
  {
    id: "supply-chain",
    label: "Supply Chain",
    subtitle: "Critical materials",
    score: 6.2,
    query:
      "China supply chain disruption risks for semiconductors, rare earths, and critical materials",
  },
  {
    id: "xi-policy",
    label: "Xi Policy",
    subtitle: "Political direction",
    score: 5.8,
    query:
      "Xi Jinping policy signals and political direction affecting China's economic and regulatory environment",
  },
  {
    id: "hong-kong",
    label: "Hong Kong",
    subtitle: "Financial stability",
    score: 5.1,
    query:
      "Hong Kong financial market stability risks and regulatory environment under Beijing's influence",
  },
];

function signalColor(score: number): string {
  if (score >= 7.5) return "#E24B4A";
  if (score >= 5.5) return "#E4A84B";
  return "#1D9E75";
}

interface SidebarProps {
  onSelectQuery: (query: string) => void;
  activeQuery?: string;
}

export default function Sidebar({ onSelectQuery, activeQuery }: SidebarProps) {
  return (
    <aside className="flex flex-col h-full overflow-y-auto bg-surface">
      <div className="px-3 py-2.5 border-b border-surface-high shrink-0">
        <p className="text-[9px] font-mono text-ink-muted uppercase tracking-widest">
          Active Signals
        </p>
      </div>

      <div className="flex flex-col flex-1 py-1 overflow-y-auto">
        {SIGNALS.map((sig) => {
          const isActive = activeQuery === sig.query;
          const color = signalColor(sig.score);
          return (
            <button
              key={sig.id}
              onClick={() => onSelectQuery(sig.query)}
              className={`w-full text-left px-3 py-2.5 transition-all border-l-2 ${
                isActive
                  ? "border-accent-teal bg-surface-mid"
                  : "border-transparent hover:bg-surface-mid hover:border-ink-muted/30"
              }`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-medium text-ink-primary truncate leading-tight">
                  {sig.label}
                </span>
                <span
                  className="text-xs font-mono font-bold shrink-0 ml-2"
                  style={{ color }}
                >
                  {sig.score.toFixed(1)}
                </span>
              </div>
              <p className="text-[10px] text-ink-muted mb-1.5 leading-tight">
                {sig.subtitle}
              </p>
              <div className="h-[3px] rounded-full overflow-hidden bg-surface-high">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(sig.score / 10) * 100}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
