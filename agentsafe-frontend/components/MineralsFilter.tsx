"use client";

export interface FilterState {
  mineral: string[];
  region: string[];
  risk_type: string[];
  severity: string; // '' = any
}

export const EMPTY_FILTERS: FilterState = {
  mineral: [],
  region: [],
  risk_type: [],
  severity: "",
};

export function activeFilterCount(f: FilterState): number {
  return (
    f.mineral.length +
    f.region.length +
    f.risk_type.length +
    (f.severity ? 1 : 0)
  );
}

const FILTER_OPTIONS = {
  mineral: [
    { value: "cobalt", label: "Cobalt" },
    { value: "lithium", label: "Lithium" },
    { value: "copper", label: "Copper" },
    { value: "rare_earth", label: "Rare Earth" },
    { value: "nickel", label: "Nickel" },
    { value: "pgm", label: "PGM" },
  ],
  region: [
    { value: "africa", label: "Africa" },
    { value: "latam", label: "LatAm" },
    { value: "drc", label: "DRC" },
    { value: "zambia", label: "Zambia" },
    { value: "chile", label: "Chile" },
    { value: "argentina", label: "Argentina" },
    { value: "peru", label: "Peru" },
    { value: "bolivia", label: "Bolivia" },
  ],
  risk_type: [
    { value: "political", label: "Political" },
    { value: "labor", label: "Labor" },
    { value: "logistics", label: "Logistics" },
    { value: "regulatory", label: "Regulatory" },
    { value: "security", label: "Security" },
    { value: "chinese_activity", label: "Chinese Activity" },
  ],
  severity: [
    { value: "high", label: "High" },
    { value: "medium", label: "Medium" },
    { value: "low", label: "Low" },
  ],
};

const SEVERITY_COLORS: Record<string, string> = {
  high: "#E24B4A",
  medium: "#E4A84B",
  low: "#1D9E75",
};

interface MineralsFilterProps {
  filters: FilterState;
  onChange: (f: FilterState) => void;
  isOpen: boolean;
  onToggleOpen: () => void;
}

function CheckRow({
  label,
  checked,
  onToggle,
}: {
  label: string;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className="flex items-center gap-2 group text-left w-full"
    >
      <span
        className={`w-3.5 h-3.5 rounded border shrink-0 flex items-center justify-center transition-all ${
          checked
            ? "bg-accent-teal border-accent-teal"
            : "border-surface-high group-hover:border-accent-teal/50"
        }`}
      >
        {checked && (
          <svg viewBox="0 0 8 6" fill="none" className="w-2 h-2">
            <polyline
              points="1,3 3,5 7,1"
              stroke="#0B0F1A"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </span>
      <span
        className={`text-xs transition-colors ${
          checked
            ? "text-ink-primary"
            : "text-ink-muted group-hover:text-ink-secondary"
        }`}
      >
        {label}
      </span>
    </button>
  );
}

function RadioRow({
  label,
  value,
  selected,
  onSelect,
}: {
  label: string;
  value: string;
  selected: string;
  onSelect: (v: string) => void;
}) {
  const isChecked = selected === value;
  const color = SEVERITY_COLORS[value] ?? "#9BA8C0";
  return (
    <button
      onClick={() => onSelect(isChecked ? "" : value)}
      className="flex items-center gap-2 group text-left w-full"
    >
      <span
        className="w-3.5 h-3.5 rounded-full border shrink-0 flex items-center justify-center transition-all"
        style={
          isChecked
            ? { borderColor: color }
            : { borderColor: "var(--surface-high)" }
        }
      >
        {isChecked && (
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: color }}
          />
        )}
      </span>
      <span
        className="text-xs transition-colors"
        style={
          isChecked
            ? { color }
            : { color: "var(--ink-muted)" }
        }
      >
        {label}
      </span>
    </button>
  );
}

function FilterGroup({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-[10px] font-mono text-ink-muted uppercase tracking-widest border-b border-surface-high pb-1.5">
        {title}
      </p>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

export default function MineralsFilter({
  filters,
  onChange,
  isOpen,
  onToggleOpen,
}: MineralsFilterProps) {
  const count = activeFilterCount(filters);

  const toggleMulti = (
    key: "mineral" | "region" | "risk_type",
    value: string
  ) => {
    const current = filters[key];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    onChange({ ...filters, [key]: next });
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Toolbar */}
      <div className="flex items-center gap-2">
        <button
          onClick={onToggleOpen}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono transition-all ${
            isOpen || count > 0
              ? "border-accent-teal/50 text-accent-teal bg-accent-teal/10"
              : "border-surface-high text-ink-muted hover:border-ink-muted hover:text-ink-secondary"
          }`}
        >
          {/* Funnel icon */}
          <svg
            viewBox="0 0 16 14"
            className="w-3.5 h-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M1 1h14l-5.5 6.5V13l-3-1.5V7.5L1 1z" />
          </svg>
          Filters{count > 0 ? ` (${count})` : ""}
        </button>

        {count > 0 && (
          <button
            onClick={() => onChange(EMPTY_FILTERS)}
            className="text-xs text-ink-muted hover:text-accent-red transition-colors font-mono"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Expandable panel */}
      {isOpen && (
        <div className="bg-surface-mid border border-surface-high rounded-xl p-4 animate-fade-up">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            <FilterGroup title="Mineral">
              {FILTER_OPTIONS.mineral.map((opt) => (
                <CheckRow
                  key={opt.value}
                  label={opt.label}
                  checked={filters.mineral.includes(opt.value)}
                  onToggle={() => toggleMulti("mineral", opt.value)}
                />
              ))}
            </FilterGroup>

            <FilterGroup title="Region">
              {FILTER_OPTIONS.region.map((opt) => (
                <CheckRow
                  key={opt.value}
                  label={opt.label}
                  checked={filters.region.includes(opt.value)}
                  onToggle={() => toggleMulti("region", opt.value)}
                />
              ))}
            </FilterGroup>

            <FilterGroup title="Risk Type">
              {FILTER_OPTIONS.risk_type.map((opt) => (
                <CheckRow
                  key={opt.value}
                  label={opt.label}
                  checked={filters.risk_type.includes(opt.value)}
                  onToggle={() => toggleMulti("risk_type", opt.value)}
                />
              ))}
            </FilterGroup>

            <FilterGroup title="Severity">
              {FILTER_OPTIONS.severity.map((opt) => (
                <RadioRow
                  key={opt.value}
                  label={opt.label}
                  value={opt.value}
                  selected={filters.severity}
                  onSelect={(v) => onChange({ ...filters, severity: v })}
                />
              ))}
            </FilterGroup>
          </div>
        </div>
      )}
    </div>
  );
}
