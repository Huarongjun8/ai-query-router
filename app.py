import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import PyPDF2
import docx
import pandas as pd
from PIL import Image
import io
import base64
import json
from datetime import datetime, timedelta
from html import escape as html_escape

# Remove top padding and hide menu for mobile
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.5rem;
            padding-bottom: 0rem;
        }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {visibility: hidden;}
        .styles_viewerBadge__1yB5_ {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Initialize API clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize rate limiting in session state
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
    st.session_state.reset_time = datetime.now() + timedelta(hours=1)

# Initialize cumulative savings tracker
if 'total_savings' not in st.session_state:
    st.session_state.total_savings = 0.0
    st.session_state.total_queries_optimized = 0
    st.session_state.total_before_cost = 0.0

st.markdown("**AI Should Be Free!**")

# Show cumulative savings badge if any queries have been optimized
if st.session_state.total_queries_optimized > 0:
    total = st.session_state.total_savings
    queries = st.session_state.total_queries_optimized
    total_before = st.session_state.total_before_cost
    pct = (total / total_before * 100) if total_before > 0 else 0
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #1a2332;border-radius:20px;'
        f'padding:0.35rem 0.8rem;font-size:0.75rem;color:#10b981;display:inline-block;'
        f'margin-bottom:0.5rem;">'
        f'⚡ {queries} queries optimized · saved ~${total:.4f} ({pct:.0f}%)'
        f'</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# RULE-BASED PROMPT ENRICHMENT — Zero cost, zero latency
# ──────────────────────────────────────────────

ENRICHMENT_RULES = {
    "invest": {
        "triggers": ["invest", "investment", "stock", "portfolio", "fund", "etf", "bond",
                      "retirement", "401k", "ira", "dividend", "equity", "asset", "allocation",
                      "return", "yield", "valuation", "p/e", "market cap", "ticker"],
        "qualifiers": [
            (["amount", "budget", "how much", "$", "dollar", "capital", "savings", "allocation"],
             "Consider: investment amount/capital available."),
            (["risk", "safe", "aggressive", "conservative", "volatile", "hedge", "downside"],
             "Consider: risk tolerance."),
            (["timeline", "term", "long", "short", "year", "retire", "goal", "horizon", "hold"],
             "Consider: investment timeline/horizon."),
            (["goal", "purpose", "why", "retirement", "house", "education", "income", "growth", "preservation"],
             "Consider: investment goals (growth, income, preservation)."),
            (["sector", "industry", "tech", "healthcare", "energy", "finance", "real estate"],
             "Consider: sector/industry focus."),
        ]
    },
    "diligence": {
        "triggers": ["due diligence", "diligence", "evaluate company", "assess", "vet",
                      "should we acquire", "acquisition target", "merger", "m&a",
                      "is this a good", "worth investing", "red flag", "risk assessment",
                      "acquir", "buyer", "suitor", "takeover", "exit", "who would buy",
                      "who could buy", "sell the company", "strategic buyer", "pe buyer",
                      "lbo", "buyout", "target company", "potential buyer"],
        "qualifiers": [
            (["revenue", "financials", "earnings", "profit", "margin", "cash flow", "balance sheet"],
             "Consider: financial health (revenue, margins, cash flow)."),
            (["market", "tam", "sam", "competitor", "moat", "market share", "industry"],
             "Consider: market position and competitive landscape."),
            (["team", "management", "founder", "ceo", "leadership", "board"],
             "Consider: management team and track record."),
            (["risk", "liability", "legal", "regulatory", "compliance", "debt"],
             "Consider: key risks (legal, regulatory, financial)."),
            (["valuation", "multiple", "price", "deal terms", "structure"],
             "Consider: valuation and deal structure."),
        ]
    },
    "financial_analysis": {
        "triggers": ["financial model", "dcf", "forecast", "projection", "pro forma",
                      "cap table", "unit economics", "burn rate", "runway", "ltv", "cac",
                      "arpu", "arr", "mrr", "gross margin", "ebitda", "roi"],
        "qualifiers": [
            (["assumption", "growth rate", "discount", "wacc", "terminal"],
             "Consider: key assumptions and growth rates."),
            (["period", "year", "quarter", "month", "timeframe", "horizon"],
             "Consider: projection timeframe."),
            (["benchmark", "comparable", "peer", "industry average", "comp"],
             "Consider: industry benchmarks and comparables."),
            (["sensitivity", "scenario", "best case", "worst case", "base case"],
             "Consider: scenario analysis (base/bull/bear case)."),
        ]
    },
    "fundraise": {
        "triggers": ["fundraise", "raise capital", "raise money", "funding round", "series a",
                      "series b", "seed round", "pitch deck", "investor", "vc", "angel",
                      "term sheet", "convertible note", "safe", "pre-money", "post-money"],
        "qualifiers": [
            (["amount", "how much", "target", "$", "million", "round size"],
             "Consider: target raise amount."),
            (["stage", "seed", "series", "pre-seed", "growth", "late stage"],
             "Consider: company stage."),
            (["use of funds", "purpose", "allocat", "spend", "deploy"],
             "Consider: use of funds / capital allocation plan."),
            (["traction", "revenue", "users", "growth", "metric", "kpi"],
             "Consider: current traction and key metrics."),
            (["timeline", "when", "close", "deadline", "runway"],
             "Consider: fundraising timeline and current runway."),
        ]
    },
    "strategy": {
        "triggers": ["strategy", "strategic", "go to market", "gtm", "market entry",
                      "competitive analysis", "positioning", "differentiat", "moat",
                      "business model", "pivot", "roadmap", "okr", "kpi"],
        "qualifiers": [
            (["market", "tam", "sam", "som", "segment", "target", "audience", "icp"],
             "Consider: target market and customer segment."),
            (["competitor", "landscape", "alternative", "incumbent", "threat"],
             "Consider: competitive landscape."),
            (["budget", "resource", "constraint", "headcount", "bandwidth"],
             "Consider: resource constraints (budget, team, time)."),
            (["timeline", "phase", "quarter", "milestone", "deadline", "when"],
             "Consider: timeline and key milestones."),
            (["metric", "success", "kpi", "goal", "measure", "target", "okr"],
             "Consider: success metrics and goals."),
        ]
    },
    "code": {
        "triggers": ["code", "coding", "program", "script", "function", "implement",
                      "build an app", "build a site", "develop", "deploy", "architect",
                      "api", "endpoint", "database", "backend", "frontend", "fullstack"],
        "qualifiers": [
            (["language", "python", "javascript", "java", "rust", "go", "typescript", "c++", "react", "node"],
             "Consider: programming language/framework."),
            (["scale", "user", "traffic", "load", "performance", "concurrent", "latency"],
             "Consider: scale and performance requirements."),
            (["stack", "aws", "gcp", "azure", "cloud", "docker", "kubernetes", "infra"],
             "Consider: infrastructure/cloud platform."),
            (["security", "auth", "encrypt", "compliance", "gdpr", "soc2", "hipaa"],
             "Consider: security and compliance requirements."),
        ]
    },
    "debug": {
        "triggers": ["fix", "debug", "error", "bug", "broken", "not working", "crash",
                      "issue", "problem with", "troubleshoot", "failing", "exception",
                      "stack trace", "won't start", "slow", "timeout", "memory leak"],
        "qualifiers": [
            (["language", "python", "javascript", "java", "react", "node", "sql", "version"],
             "Consider: language/framework and version."),
            (["error", "message", "traceback", "stack trace", "log", "output", "symptom"],
             "Consider: exact error message or symptoms."),
            (["tried", "already", "attempted", "so far", "done"],
             "Consider: what has already been tried."),
            (["environment", "local", "production", "staging", "docker", "os", "browser"],
             "Consider: environment (local/production/OS/browser)."),
        ]
    },
    "architecture": {
        "triggers": ["architect", "system design", "infrastructure", "microservice", "monolith",
                      "tech stack", "migration", "refactor", "redesign", "scalab", "data pipeline",
                      "etl", "event driven", "message queue", "cache"],
        "qualifiers": [
            (["scale", "user", "request", "rps", "qps", "traffic", "data volume", "tb", "gb"],
             "Consider: scale requirements (users, data volume, throughput)."),
            (["budget", "cost", "spend", "$", "optimize", "reduce"],
             "Consider: infrastructure budget constraints."),
            (["team", "engineer", "headcount", "maintain", "complexity"],
             "Consider: team size and maintenance capacity."),
            (["latency", "uptime", "sla", "availability", "reliability", "99.9"],
             "Consider: reliability and latency requirements."),
            (["existing", "legacy", "current", "migrate from", "already using"],
             "Consider: existing tech stack and migration constraints."),
        ]
    },
    "hire": {
        "triggers": ["hire", "recruit", "find someone", "looking for a", "contractor",
                      "freelancer", "employee", "headcount", "job description", "jd",
                      "interview", "talent", "staffing", "outsource"],
        "qualifiers": [
            (["budget", "salary", "rate", "cost", "pay", "comp", "$", "hourly", "annual", "equity"],
             "Consider: compensation range and equity."),
            (["location", "remote", "onsite", "hybrid", "city", "local", "where", "timezone"],
             "Consider: location requirements (remote/onsite/hybrid/timezone)."),
            (["when", "start", "timeline", "urgent", "asap", "deadline", "duration", "contract", "perm"],
             "Consider: start date and contract type (perm/contract)."),
            (["skill", "experience", "senior", "junior", "qualification", "years", "stack"],
             "Consider: required skills and experience level."),
            (["culture", "team", "report", "manage", "fit"],
             "Consider: team dynamics and reporting structure."),
        ]
    },
    "write": {
        "triggers": ["write", "draft", "compose", "memo", "email", "blog", "post",
                      "article", "newsletter", "announcement", "press release", "copy",
                      "report", "brief", "summary", "executive summary", "board update",
                      "investor update", "earnings"],
        "qualifiers": [
            (["audience", "reader", "who", "recipient", "stakeholder", "board", "investor"],
             "Consider: target audience."),
            (["tone", "formal", "casual", "professional", "friendly", "urgent", "persuasive"],
             "Consider: tone and voice."),
            (["length", "word", "page", "short", "long", "brief", "detailed", "concise"],
             "Consider: desired length."),
            (["purpose", "goal", "action", "cta", "inform", "persuade", "update", "request"],
             "Consider: goal/desired action from the reader."),
        ]
    },
    "sell": {
        "triggers": ["sell", "sales", "pitch", "outreach", "cold email", "prospect",
                      "lead gen", "pipeline", "close deal", "proposal", "rfp", "partnership",
                      "business development", "bd"],
        "qualifiers": [
            (["target", "audience", "buyer", "persona", "icp", "decision maker", "who"],
             "Consider: target buyer/decision maker."),
            (["product", "service", "offering", "solution", "value prop"],
             "Consider: product/service being sold."),
            (["deal size", "contract", "price", "budget", "$", "arr", "acv"],
             "Consider: deal size and pricing."),
            (["stage", "cold", "warm", "follow up", "closing", "negotiat"],
             "Consider: sales stage (cold outreach vs follow-up vs closing)."),
            (["timeline", "quarter", "deadline", "end of", "close by"],
             "Consider: deal timeline/urgency."),
        ]
    },
    "analyze": {
        "triggers": ["analyze", "analysis", "data", "dataset", "metrics", "dashboard",
                      "report on", "trend", "insight", "visualization", "chart", "graph",
                      "sql", "query", "tableau", "looker", "excel"],
        "qualifiers": [
            (["source", "database", "csv", "api", "warehouse", "table", "spreadsheet"],
             "Consider: data source and format."),
            (["timeframe", "period", "date range", "last quarter", "ytd", "yoy", "mom"],
             "Consider: time period for analysis."),
            (["metric", "kpi", "measure", "dimension", "breakout", "segment"],
             "Consider: key metrics and dimensions to analyze."),
            (["audience", "who", "stakeholder", "present", "board", "team", "exec"],
             "Consider: who will consume this analysis."),
            (["action", "decision", "recommend", "so what", "takeaway", "insight"],
             "Consider: what decision this analysis should inform."),
        ]
    },
    "product": {
        "triggers": ["product", "feature", "launch", "ship", "release", "mvp", "prototype",
                      "user story", "prd", "spec", "requirements", "roadmap", "backlog",
                      "prioritize", "sprint"],
        "qualifiers": [
            (["user", "customer", "persona", "segment", "who is this for"],
             "Consider: target user/customer."),
            (["problem", "pain point", "need", "job to be done", "why"],
             "Consider: problem being solved."),
            (["timeline", "deadline", "quarter", "sprint", "milestone", "when"],
             "Consider: launch timeline."),
            (["constraint", "resource", "budget", "team", "technical", "dependency"],
             "Consider: constraints (technical, resource, budget)."),
            (["success", "metric", "kpi", "measure", "goal", "adoption"],
             "Consider: success metrics."),
        ]
    },
    "compare": {
        "triggers": ["compare", "versus", " vs ", "difference between", "better",
                      "which is", "which should", "pros and cons", "tradeoff",
                      "evaluate", "benchmark", "alternative"],
        "qualifiers": [
            (["budget", "cost", "price", "value", "afford", "$", "worth", "tco"],
             "Consider: cost/TCO comparison."),
            (["priority", "important", "matter", "care about", "need", "must have", "criteria", "weight"],
             "Consider: key decision criteria and weights."),
            (["use case", "purpose", "for", "scenario", "situation", "need it for", "context"],
             "Consider: specific use case and context."),
            (["scale", "team size", "volume", "growth", "user"],
             "Consider: scale and growth requirements."),
        ]
    },
    "legal": {
        "triggers": ["legal", "lawyer", "contract", "agreement", "terms", "compliance",
                      "regulation", "sec", "filing", "10-k", "10-q", "proxy", "audit",
                      "ip", "patent", "trademark", "nda", "liability", "sue", "dispute"],
        "qualifiers": [
            (["jurisdiction", "state", "country", "us", "china", "eu", "cross-border"],
             "Consider: jurisdiction (US, China, cross-border, etc.)."),
            (["type", "kind", "nature", "category", "area of law"],
             "Consider: type of legal matter."),
            (["urgency", "deadline", "filing date", "when", "timeline", "statute"],
             "Consider: timeline and deadlines."),
            (["budget", "cost", "afford", "$", "fee", "retainer"],
             "Consider: budget for legal services."),
        ]
    },
    "find": {
        "triggers": ["find", "search", "look for", "locate", "recommend", "suggest",
                      "best", "top", "options for", "where can"],
        "qualifiers": [
            (["near", "nearby", "local", "area", "city", "location", "where", "remote"],
             "Consider: location/proximity."),
            (["criteria", "requirement", "specific", "prefer", "feature", "must have"],
             "Consider: specific criteria or requirements."),
            (["budget", "price", "cost", "free", "cheap", "$", "enterprise"],
             "Consider: budget constraints."),
        ]
    },
    "plan": {
        "triggers": ["plan", "organize", "schedule", "arrange", "prepare", "set up",
                      "host", "offsite", "retreat", "conference", "meeting"],
        "qualifiers": [
            (["budget", "cost", "price", "afford", "$", "spend"],
             "Consider: budget."),
            (["people", "person", "guest", "attendee", "group", "team", "how many", "headcount"],
             "Consider: number of people involved."),
            (["when", "date", "time", "timeline", "deadline", "month", "week", "day", "quarter"],
             "Consider: dates/timeline."),
            (["where", "location", "venue", "place", "indoor", "outdoor", "virtual", "online", "city"],
             "Consider: location/venue."),
        ]
    },
    "learn": {
        "triggers": ["learn", "study", "understand", "teach me", "how to", "tutorial",
                      "course", "explain", "master", "get better at", "upskill"],
        "qualifiers": [
            (["beginner", "advanced", "intermediate", "expert", "new to", "experience", "level", "basics"],
             "Consider: current skill/knowledge level."),
            (["time", "quick", "fast", "hour", "week", "month", "schedule", "pace"],
             "Consider: time commitment and learning pace."),
            (["goal", "purpose", "why", "career", "job", "project", "certification", "interview"],
             "Consider: learning goal or purpose."),
        ]
    },
    "travel": {
        "triggers": ["travel", "trip", "vacation", "visit", "fly", "flight", "hotel",
                      "itinerary", "tour", "safari", "book a", "cruise", "getaway", "resort"],
        "qualifiers": [
            (["budget", "cost", "cheap", "luxury", "afford", "$", "price"],
             "Consider: budget range."),
            (["when", "date", "month", "season", "how long", "days", "week", "duration"],
             "Consider: travel dates and duration."),
            (["people", "solo", "couple", "family", "group", "kids", "children", "friend"],
             "Consider: who is traveling."),
            (["interest", "like", "enjoy", "adventure", "relax", "culture", "food", "beach", "mountain"],
             "Consider: interests and activity preferences."),
        ]
    },
    "cross_border": {
        "triggers": ["china", "chinese", "cross-border", "us-china", "localize", "translate",
                      "wfoe", "vie", "hkex", "shanghai", "shenzhen", "nasdaq ipo",
                      "going global", "enter china", "enter us market", "sec filing",
                      "chinese company", "expand to"],
        "qualifiers": [
            (["structure", "entity", "vie", "wfoe", "jv", "subsidiary", "holdco"],
             "Consider: corporate/entity structure."),
            (["regulatory", "compliance", "sec", "csrc", "cfius", "data", "cybersecurity"],
             "Consider: regulatory and compliance requirements."),
            (["language", "mandarin", "english", "bilingual", "translate"],
             "Consider: language and localization needs."),
            (["timeline", "when", "phase", "milestone", "deadline"],
             "Consider: timeline and phasing."),
            (["budget", "cost", "capital", "$", "rmb", "usd", "forex", "repatriat"],
             "Consider: budget and currency/FX considerations."),
        ]
    },
}


def enrich_prompt(user_prompt):
    prompt_lower = user_prompt.lower()
    additions = []
    matched_categories = []
    for category, rules in ENRICHMENT_RULES.items():
        for trigger in rules["triggers"]:
            if trigger in prompt_lower:
                matched_categories.append(category)
                break
    if not matched_categories:
        return user_prompt, []
    for category in matched_categories:
        rules = ENRICHMENT_RULES[category]
        for check_words, addition_text in rules["qualifiers"]:
            already_mentioned = any(word in prompt_lower for word in check_words)
            if not already_mentioned:
                if addition_text not in additions:
                    additions.append(addition_text)
    if not additions:
        return user_prompt, []
    enriched = user_prompt.rstrip() + "\n\n[Auto-enriched context: " + " ".join(additions) + "]"
    return enriched, additions


# ──────────────────────────────────────────────
# PROMPT OPTIMIZER
# ──────────────────────────────────────────────
OPTIMIZER_MODEL = "llama-3.3-70b-versatile"

OPTIMIZER_SYSTEM_PROMPT = """You are a prompt optimization engine. Take the user's prompt and make it MORE EFFECTIVE and LESS EXPENSIVE to run.

Return ONLY valid JSON:
{
  "optimized_prompt": "The improved prompt text",
  "changes_summary": "One short sentence describing what you improved",
  "complexity": "low|medium|high",
  "task_type": "classification|generation|analysis|conversation|coding|translation|summarization|other",
  "recommended_model_tier": "budget|mid|premium",
  "tier_reason": "One sentence on why this tier"
}

RULES:
1. REMOVE filler, redundancy, unnecessary politeness
2. ADD specificity and output format constraints where helpful
3. RESTRUCTURE for clarity if needed
4. PRESERVE the user's intent exactly — never change what they're asking for
5. Simple tasks (lookups, short answers, chat) → "budget"
6. Moderate reasoning, writing, analysis → "mid"
7. Complex coding, nuanced creative, multi-step reasoning → "premium"
8. If the prompt is already clean and tight, return it mostly unchanged

Return ONLY valid JSON. No markdown fences, no commentary."""


def optimize_prompt(user_prompt):
    try:
        response = groq_client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {
            "optimized_prompt": user_prompt,
            "changes_summary": "Optimizer returned invalid format — using original",
            "complexity": "medium",
            "task_type": "other",
            "recommended_model_tier": "mid",
        }
    except Exception:
        return {
            "optimized_prompt": user_prompt,
            "changes_summary": "Optimizer unavailable — using original",
            "complexity": "medium",
            "task_type": "other",
            "recommended_model_tier": "mid",
        }


def render_optimization_card(result):
    """Show optimization card using FLAT HTML only.
    Streamlit's markdown parser breaks on nested <div> elements,
    causing closing </div> tags to leak as visible text.
    Fix: use only <span> and <br> inside a single outer <div>.
    """
    tier = result.get("model_tier", "mid")
    tier_colors = {"budget": "#10b981", "mid": "#f59e0b", "premium": "#8b5cf6"}
    color = tier_colors.get(tier, "#94a3b8")

    savings_pct = result.get("savings_pct", 0)
    savings_display = html_escape(f"-{savings_pct:.0f}% cost" if savings_pct > 1 else "optimal")

    changed = result.get("changed", False)
    enrichment = result.get("enrichment_additions", [])

    # Escape ALL dynamic content to prevent HTML injection
    model_label = html_escape(str(result.get("model_label", "")))
    changes_summary = html_escape(str(result.get("changes_summary", "")))

    # Build detail lines with <span> + <br> only — NO nested divs
    detail_lines = ""
    if changed:
        detail_lines += (
            f'<br><span style="color:#8b949e;font-size:0.78rem;">'
            f'✏️ {changes_summary}</span>'
        )
    else:
        detail_lines += (
            f'<br><span style="color:#8b949e;font-size:0.78rem;">'
            f'✓ Prompt already clean — no changes needed</span>'
        )

    if enrichment:
        enriched_items = ", ".join(html_escape(str(e).replace("Consider: ", "")) for e in enrichment)
        detail_lines += (
            f'<br><span style="color:#a78bfa;font-size:0.78rem;">'
            f'🧠 Auto-enriched: {enriched_items}</span>'
        )

    # Single flat <div> — zero nesting
    card_html = (
        f'<div style="'
        f'background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);'
        f'border-left:3px solid {color};'
        f'border-radius:8px;'
        f'padding:0.7rem 0.9rem;'
        f'margin:0.3rem 0 0.6rem 0;'
        f'font-size:0.82rem;'
        f'">'
        f'<span style="color:{color};font-weight:700;font-size:0.73rem;'
        f'text-transform:uppercase;letter-spacing:0.04em;">'
        f'⚡ Optimized → {model_label}</span>'
        f' <span style="color:#10b981;font-weight:600;font-size:0.78rem;float:right;">'
        f'{savings_display}</span>'
        f'{detail_lines}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def est_tokens(text):
    return max(1, len(text) // 4)


def est_cost(prompt, cost_per_1m, output_tokens=500):
    input_tokens = est_tokens(prompt)
    return ((input_tokens + output_tokens) / 1_000_000) * cost_per_1m


# ──────────────────────────────────────────────
# FILE PROCESSING FUNCTIONS
# ──────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return "Error reading PDF: " + str(e)

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return "Error reading DOCX: " + str(e)

def process_csv_excel(uploaded_file, file_type):
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        summary = "Dataset with " + str(len(df)) + " rows and " + str(len(df.columns)) + " columns.\n\n"
        summary += "Columns: " + ", ".join(df.columns.tolist()) + "\n\n"
        summary += "First few rows:\n" + df.head().to_string() + "\n\n"
        summary += "Data types:\n" + df.dtypes.to_string() + "\n\n"
        summary += "Basic statistics:\n" + df.describe().to_string()
        return summary
    except Exception as e:
        return "Error reading " + file_type.upper() + ": " + str(e)

def image_to_base64(image_file):
    try:
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format=image.format if image.format else "PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        media_type = "image/png"
        if image.format == "JPEG":
            media_type = "image/jpeg"
        elif image.format == "WEBP":
            media_type = "image/webp"
        elif image.format == "GIF":
            media_type = "image/gif"
        return img_str, media_type
    except Exception as e:
        return None, "Error processing image: " + str(e)

# COMBINED INPUT AREA
query = st.text_area(
    "💬 Enter your query:",
    height=100,
    placeholder="Ask anything or ask about your uploaded files..."
)

with st.expander("📎 Attach files (optional)"):
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx", "csv", "xlsx", "png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
        help="Drag and drop or click to upload",
        label_visibility="collapsed"
    )

if uploaded_files:
    st.caption("✅ " + str(len(uploaded_files)) + " file(s) attached")

# Sidebar rate limit
st.sidebar.write("🔄 Queries used: " + str(st.session_state.query_count) + "/20 this hour")
st.sidebar.caption("Resets at: " + st.session_state.reset_time.strftime('%I:%M %p'))

# Process uploaded files
file_contents = []
image_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == "pdf":
            content = extract_text_from_pdf(uploaded_file)
            file_contents.append("[PDF: " + uploaded_file.name + "]\n" + content + "\n")
        elif file_type == "txt":
            content = uploaded_file.read().decode('utf-8')
            file_contents.append("[Text file: " + uploaded_file.name + "]\n" + content + "\n")
        elif file_type == "docx":
            content = extract_text_from_docx(uploaded_file)
            file_contents.append("[Word document: " + uploaded_file.name + "]\n" + content + "\n")
        elif file_type in ["csv", "xlsx"]:
            content = process_csv_excel(uploaded_file, file_type)
            file_contents.append("[Spreadsheet: " + uploaded_file.name + "]\n" + content + "\n")
        elif file_type in ["png", "jpg", "jpeg", "webp", "gif"]:
            img_base64, media_type = image_to_base64(uploaded_file)
            if img_base64:
                image_data.append({
                    "name": uploaded_file.name,
                    "base64": img_base64,
                    "media_type": media_type
                })
                with st.expander("🖼️ Preview: " + uploaded_file.name):
                    st.image(uploaded_file, width=300)
            else:
                st.error(media_type)

# Routing logic
def route_query(query, has_files=False, has_images=False):
    query_lower = query.lower()
    word_count = len(query.split())
    if has_images:
        return "claude-sonnet-4-20250514", "Claude Sonnet 4 (Image analysis)", 0.003, "anthropic"
    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy', 'summarize']
    complexity_score = sum(1 for keyword in complex_keywords if keyword in query_lower)
    if has_files:
        complexity_score += 1
    if complexity_score >= 2 or word_count > 50:
        return "claude-sonnet-4-20250514", "Claude Sonnet 4 (Complex reasoning)", 0.003, "anthropic"
    elif word_count > 20 or complexity_score == 1:
        return "llama-3.3-70b-versatile", "Groq Llama 3.3 70B (Fast & free!)", 0.00059, "groq"
    else:
        return "llama-3.1-8b-instant", "Groq Llama 3.1 8B (Open source & fast!)", 0.00005, "groq"

# Mode selection
mode = st.radio("Routing mode:", ["Auto (Recommended)", "Manual Override"])

if mode == "Manual Override":
    model_choice = st.selectbox("Choose model:",
        ["Llama 3.1 8B (Open Source - Fast)", "Groq Llama 3.3 70B - Fast & Cheap", "GPT-4o Mini", "Claude Sonnet 4"])

# Submit button
if st.button("🚀 Send Query", type="primary", use_container_width=True):
    if query or uploaded_files:
        if st.session_state.query_count >= 20:
            if datetime.now() < st.session_state.reset_time:
                st.error("⏱️ Rate limit reached! You've used " + str(st.session_state.query_count) + "/20 queries this hour.")
                st.info("Your limit resets at " + st.session_state.reset_time.strftime('%I:%M %p') + ". Please try again then!")
                st.caption("Rate limits help us keep the service free and sustainable. Thank you for understanding!")
                st.stop()
            else:
                st.session_state.query_count = 0
                st.session_state.reset_time = datetime.now() + timedelta(hours=1)

        with st.spinner("⚡ Optimizing and routing..."):
            enriched_query, enrichment_additions = enrich_prompt(query)
            opt_result = optimize_prompt(enriched_query)
            optimized_query = opt_result.get("optimized_prompt", enriched_query)
            changed = query.strip().lower() != optimized_query.strip().lower()

            baseline_cost_per_1m = 15.0
            before_cost = est_cost(query, baseline_cost_per_1m)

            full_query = optimized_query
            if file_contents:
                full_query = "\n\n".join(file_contents) + "\n\nUser question: " + optimized_query

            if mode == "Auto (Recommended)":
                model_id, model_name, cost_per_1m, provider = route_query(
                    optimized_query,
                    has_files=len(file_contents) > 0,
                    has_images=len(image_data) > 0
                )
                after_cost = est_cost(optimized_query, cost_per_1m)
                savings_pct = ((before_cost - after_cost) / before_cost * 100) if before_cost > 0 else 0

                opt_card = {
                    "model_tier": opt_result.get("recommended_model_tier", "mid"),
                    "model_label": model_name,
                    "savings_pct": savings_pct,
                    "changed": changed,
                    "changes_summary": opt_result.get("changes_summary", ""),
                    "optimized": optimized_query,
                    "enrichment_additions": enrichment_additions,
                }

        if mode == "Auto (Recommended)":
            render_optimization_card(opt_card)
            st.session_state.total_savings += (before_cost - after_cost)
            st.session_state.total_queries_optimized += 1
            st.session_state.total_before_cost += before_cost
            st.info("🎯 Routed to: **" + model_name + "**")
        else:
            manual_opt_card = {
                "model_tier": opt_result.get("recommended_model_tier", "mid"),
                "model_label": model_choice,
                "savings_pct": 0,
                "changed": changed,
                "changes_summary": opt_result.get("changes_summary", ""),
                "optimized": optimized_query,
                "enrichment_additions": enrichment_additions,
            }
            render_optimization_card(manual_opt_card)

        with st.spinner("Generating response..."):
            try:
                if mode == "Auto (Recommended)":

                    query_lower = optimized_query.lower()
                    word_count = len(optimized_query.split())
                    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy', 'summarize']
                    complexity_score = sum(1 for keyword in complex_keywords if keyword in query_lower)

                    if len(image_data) > 0:
                        confidence = 100
                        reasoning = "Image analysis requires vision model (only Claude supports this)"
                    elif complexity_score >= 2 or word_count > 50:
                        confidence = 90 + min(complexity_score * 2, 10)
                        reasoning = "High complexity detected (" + str(complexity_score) + " complex keywords, " + str(word_count) + " words)"
                    elif word_count < 10 and complexity_score == 0:
                        confidence = 95
                        reasoning = "Simple query (" + str(word_count) + " words, no complex keywords) - using open source Llama"
                    else:
                        confidence = 70 + (complexity_score * 5)
                        reasoning = "Medium complexity (" + str(complexity_score) + " complex keywords, " + str(word_count) + " words)"

                    if file_contents:
                        reasoning += ", " + str(len(file_contents)) + " file(s) uploaded"

                    if provider == "groq":
                        response = groq_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": full_query}],
                            temperature=0.3,
                            max_tokens=2048
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens

                    elif provider == "anthropic":
                        if image_data:
                            content_parts = []
                            for img in image_data:
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": img["media_type"],
                                        "data": img["base64"]
                                    }
                                })
                            content_parts.append({"type": "text", "text": full_query})
                            response = anthropic_client.messages.create(
                                model=model_id,
                                max_tokens=2048,
                                messages=[{"role": "user", "content": content_parts}]
                            )
                        else:
                            response = anthropic_client.messages.create(
                                model=model_id,
                                max_tokens=2048,
                                messages=[{"role": "user", "content": full_query}]
                            )
                        answer = response.content[0].text
                        tokens_used = response.usage.input_tokens + response.usage.output_tokens

                    else:
                        response = openai_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": full_query}]
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens

                    st.session_state.query_count += 1

                    st.success("Response:")
                    st.write(answer)

                    estimated_cost = (tokens_used / 1000000) * cost_per_1m
                    gpt4_cost_per_1m = 0.005
                    gpt4_would_cost = (tokens_used / 1000000) * gpt4_cost_per_1m
                    savings = gpt4_would_cost - estimated_cost
                    savings_percent = (savings / gpt4_would_cost) * 100 if gpt4_would_cost > 0 else 0

                    energy_rates = {
                        "gpt-4o-mini": 0.001,
                        "gpt-4o": 0.003,
                        "claude-sonnet-4-20250514": 0.002,
                        "llama-3.1-8b-instant": 0.0003,
                        "llama-3.3-70b-versatile": 0.0008
                    }

                    energy_per_1k = energy_rates.get(model_id, 0.002)
                    energy_used_wh = (tokens_used / 1000) * energy_per_1k
                    gpt4_energy_wh = (tokens_used / 1000) * 0.003
                    energy_saved_wh = gpt4_energy_wh - energy_used_wh
                    co2_grams = (energy_used_wh / 1000) * 400
                    co2_saved_grams = (energy_saved_wh / 1000) * 400

                    st.subheader("💵 Cost Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Smart Routing Cost", "$" + str(round(estimated_cost, 6)))
                    with col2:
                        st.metric("GPT-4 Would Cost", "$" + str(round(gpt4_would_cost, 6)))
                    with col3:
                        st.metric("You Saved", "$" + str(round(savings, 6)), delta="-" + str(round(savings_percent, 1)) + "%", delta_color="inverse")

                    st.subheader("🌍 Environmental Impact")
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Energy Used", str(round(energy_used_wh, 4)) + " Wh")
                    with col5:
                        st.metric("CO2 Emissions", str(round(co2_grams, 3)) + " g")
                    with col6:
                        st.metric("Energy Saved", str(round(energy_saved_wh, 4)) + " Wh", delta=str(round(co2_saved_grams, 3)) + "g CO2", delta_color="inverse")

                    st.caption("Environmental estimates are approximate based on published AI energy research")

                    with st.expander("🔍 Routing Observability & Explainability", expanded=False):
                        obs_col1, obs_col2 = st.columns(2)
                        with obs_col1:
                            st.metric("Routing Confidence", str(confidence) + "%")
                            st.write("**Decision Reasoning:**")
                            st.write("• " + reasoning)
                            st.write("• Query length: " + str(word_count) + " words")
                            st.write("• Complexity indicators: " + str(complexity_score))
                            if file_contents:
                                st.write("• Files analyzed: " + str(len(file_contents)))
                            if image_data:
                                st.write("• Images analyzed: " + str(len(image_data)))
                        with obs_col2:
                            st.write("**Alternative Models Considered:**")
                            if model_id == "llama-3.1-8b-instant":
                                st.write("✅ Llama 3.1 8B (selected) - Open source, fast!")
                                st.write("⚪ Llama 3.3 70B - Larger, unnecessary for simple queries")
                                st.write("⚪ Claude Sonnet 4 - Expensive, overkill")
                            elif provider == "groq":
                                st.write("✅ Llama 3.3 70B (selected) - Fast & cheap!")
                                st.write("⚪ Llama 3.1 8B - Smaller, less capable for medium tasks")
                                st.write("⚪ Claude Sonnet 4 - Higher cost, unnecessary")
                            else:
                                st.write("✅ Claude Sonnet 4 (selected)")
                                st.write("⚪ Llama 3.3 70B - Lower cost but less capable for complex reasoning")
                                st.write("⚪ Llama 3.1 8B - Too small for this task")
                            st.write("**Routing Algorithm:**")
                            st.write("• Rule-based prompt enrichment (zero cost)")
                            st.write("• LLM prompt optimization (via Llama 3.3 70B)")
                            st.write("• Keyword analysis")
                            st.write("• Length-based heuristics")
                            st.write("• File type detection")
                            st.write("• Cost-performance optimization")

                    with st.expander("⚡ Prompt Optimization Details", expanded=False):
                        if enrichment_additions:
                            st.write("**🧠 Rule-based enrichment (zero cost):**")
                            for addition in enrichment_additions:
                                st.write("• " + addition)
                            st.divider()
                        opt_col1, opt_col2 = st.columns(2)
                        with opt_col1:
                            st.write("**Original prompt:**")
                            st.code(query, language=None)
                            st.caption(f"~{est_tokens(query)} tokens")
                        with opt_col2:
                            st.write("**Optimized prompt:**")
                            st.code(optimized_query, language=None)
                            st.caption(f"~{est_tokens(optimized_query)} tokens")
                        if changed:
                            st.write("**Changes:** " + opt_result.get("changes_summary", ""))
                        else:
                            st.write("**Changes:** None needed — prompt was already efficient")
                        st.write("**Task type:** " + opt_result.get("task_type", "N/A"))
                        st.write("**Complexity:** " + opt_result.get("complexity", "N/A"))

                else:
                    full_query_manual = optimized_query
                    if file_contents:
                        full_query_manual = "\n\n".join(file_contents) + "\n\nUser question: " + optimized_query

                    if "Llama 3.1 8B" in model_choice:
                        response = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": full_query_manual}]
                        )
                        answer = response.choices[0].message.content
                    elif "Groq" in model_choice:
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": full_query_manual}]
                        )
                        answer = response.choices[0].message.content
                    elif "GPT" in model_choice:
                        model = "gpt-4o-mini" if "Mini" in model_choice else "gpt-4o"
                        response = openai_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": full_query_manual}]
                        )
                        answer = response.choices[0].message.content
                    else:
                        if image_data:
                            content_parts = []
                            for img in image_data:
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": img["media_type"],
                                        "data": img["base64"]
                                    }
                                })
                            content_parts.append({"type": "text", "text": full_query_manual})
                            response = anthropic_client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=2048,
                                messages=[{"role": "user", "content": content_parts}]
                            )
                        else:
                            response = anthropic_client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=2048,
                                messages=[{"role": "user", "content": full_query_manual}]
                            )
                        answer = response.content[0].text

                    st.session_state.query_count += 1
                    st.success("Response:")
                    st.write(answer)

            except Exception as e:
                st.error("Error: " + str(e))
    else:
        st.warning("Please enter a query or upload files!")

with st.expander("ℹ️ How Auto-Routing Works"):
    st.write("""
    Smart routing saves you 70-100% on AI costs AND reduces environmental impact by:
    - **🧠 Rule-Based Enrichment**: Automatically detects your intent and adds missing context — zero cost, zero latency
    - **⚡ Automatic Prompt Optimization**: Every query is refined by AI for clarity and efficiency
    - Simple queries: Llama 3.1 8B via Groq (open source, Meta)
    - Medium queries: Llama 3.3 70B via Groq (open source, fast)
    - Complex reasoning/coding: Claude Sonnet 4
    - Image analysis: Claude Sonnet 4 (vision capabilities)

    Two-Stage Optimization:
    1. Rule-based enrichment detects intent across 20 categories — runs instantly, no API call
    2. LLM optimizer removes filler and recommends the cheapest capable model — via Groq (~1 sec)

    Rate Limits:
    - 20 queries per hour per user (resets hourly)
    - Helps keep the service free and sustainable

    File Upload Features:
    - PDFs, Text/Word docs, CSV/Excel, Images (via Claude vision)

    Why Groq?
    - Lightning fast responses (often under 1 second)
    - Runs open source Meta Llama models
    - Very affordable: $0.59 per 1M tokens
    - Can achieve 95-100% savings on simple queries vs GPT-4
    """)
