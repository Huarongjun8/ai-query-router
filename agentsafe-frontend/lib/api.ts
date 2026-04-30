const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://web-production-ae1f5.up.railway.app";

export interface ArticleTags {
  mineral: string[];
  region: string[];
  risk_type: string[];
  severity: string;
}

export interface MarketSignal {
  question: string;
  probability: number;  // 0.0–1.0
  volume: number;       // USD or forecaster count
  url: string;
}

export interface GDELTSignal {
  article_count: number;
  avg_tone: number;
  most_negative_tone: number;
  top_sources: string[];
  timespan: string;
}

export interface RiskAssessment {
  title?: string;
  risk_score?: number;
  likelihood?: number;
  severity?: number;
  reversibility?: number;
  time_horizon?: string;
  affected_sectors?: string[];
  geographic_scope?: string[];
  summary?: string;
  recommendations?: string[];
  sources?: string[];
  assessed_at?: string;
  market_signals?: MarketSignal[];
  gdelt_signal?: GDELTSignal | null;
}

export interface BriefResponse {
  brief_id: string;
  status: "pending" | "processing" | "complete" | "error";
  title?: string;
  executive_summary?: string;
  // Nested assessments — the primary source of metric fields
  risk_assessments?: RiskAssessment[];
  // Top-level fallbacks (some API versions hoist these)
  risk_score?: number;
  likelihood?: number;
  severity?: number;
  reversibility?: number;
  time_horizon?: string;
  affected_sectors?: string[];
  geographic_scope?: string[];
  recommendations?: string[];
  sources?: string[];
  compliance_status?: string;
  disclaimers?: string[];
  generated_at?: string;
  market_signals?: MarketSignal[];
  gdelt_signal?: GDELTSignal | null;
  tags?: ArticleTags;
  minerals_report?: string;
  error?: string;
}

export function deriveTags(brief: BriefResponse): ArticleTags {
  const text = [
    brief.title ?? "",
    brief.executive_summary ?? "",
    ...(brief.affected_sectors ?? []),
    ...(brief.geographic_scope ?? []),
    ...(brief.risk_assessments?.flatMap((r) => [
      r.summary ?? "",
      ...(r.affected_sectors ?? []),
      ...(r.geographic_scope ?? []),
    ]) ?? []),
  ]
    .join(" ")
    .toLowerCase();

  const mineral: string[] = [];
  if (/\bcobalt\b/.test(text)) mineral.push("cobalt");
  if (/\blithium\b/.test(text)) mineral.push("lithium");
  if (/\bcopper\b/.test(text)) mineral.push("copper");
  if (/rare[\s-]?earth/.test(text)) mineral.push("rare_earth");
  if (/\bnickel\b/.test(text)) mineral.push("nickel");
  if (/\b(pgm|platinum)\b/.test(text)) mineral.push("pgm");

  const region: string[] = [];
  if (/\b(africa|african)\b/.test(text)) region.push("africa");
  if (/\b(latin\s+america|latam|south\s+america)\b/.test(text)) region.push("latam");
  if (/\b(drc|congo)\b/.test(text)) region.push("drc");
  if (/\bzambia\b/.test(text)) region.push("zambia");
  if (/\bchile\b/.test(text)) region.push("chile");
  if (/\bargentina\b/.test(text)) region.push("argentina");
  if (/\bperu\b/.test(text)) region.push("peru");
  if (/\bbolivia\b/.test(text)) region.push("bolivia");

  const risk_type: string[] = [];
  if (/\b(coup|junta|nationali\w*|expropri\w*|sovereignty)\b/.test(text)) risk_type.push("political");
  if (/\b(strike|protest|union\s+action|artisanal\s+mining)\b/.test(text)) risk_type.push("labor");
  if (/\b(port\s+(strike|closure)|rail\s+disruption|shipping\s+delay|choke\s+point|supply\s+disruption|export\s+ban)\b/.test(text)) risk_type.push("logistics");
  if (/\b(royalt\w*|windfall|mining\s+ban|moratorium|concession|regulat\w*)\b/.test(text)) risk_type.push("regulatory");
  if (/\b(armed\s+group|militia|kidnap\w*|insurgenc\w*|terror\w*|m23|adf)\b/.test(text)) risk_type.push("security");
  if (/\b(catl|zijin|ganfeng|cmoc|minmetals|chinalco)\b/.test(text)) risk_type.push("chinese_activity");

  const riskScore =
    brief.risk_assessments?.[0]?.risk_score ?? brief.risk_score ?? 0;
  const derivedSeverity =
    riskScore >= 7 ? "high" : riskScore >= 4 ? "medium" : "low";

  // Prefer backend-provided tags if present
  const bt = brief.tags;
  return {
    mineral: bt?.mineral?.length ? bt.mineral : mineral,
    region: bt?.region?.length ? bt.region : region,
    risk_type: bt?.risk_type?.length ? bt.risk_type : risk_type,
    severity: bt?.severity ?? derivedSeverity,
  };
}

export interface SubmitResponse {
  brief_id: string;
  status: string;
  message?: string;
}

export async function submitQuery(query: string): Promise<SubmitResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to submit query (${res.status}): ${text}`);
  }

  return res.json();
}

export async function getBrief(briefId: string): Promise<BriefResponse> {
  const res = await fetch(`${API_BASE}/brief/${briefId}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch brief (${res.status}): ${text}`);
  }

  return res.json();
}

export async function waitForBrief(
  briefId: string,
  onPoll?: (brief: BriefResponse) => void,
  intervalMs = 2000,
  timeoutMs = 120_000
): Promise<BriefResponse> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const brief = await getBrief(briefId);
    if (onPoll) onPoll(brief);

    if (brief.status === "complete") return brief;
    if (brief.status === "error") {
      throw new Error(brief.error || "Pipeline returned error status");
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error("Timed out waiting for brief to complete");
}
