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
        f"""<div style="
            background: #0d1117;
            border: 1px solid #1a2332;
            border-radius: 20px;
            padding: 0.35rem 0.8rem;
            font-size: 0.75rem;
            color: #10b981;
            display: inline-block;
            margin-bottom: 0.5rem;
        ">
            ⚡ {queries} queries optimized · saved ~${total:.4f} ({pct:.0f}%)
        </div>""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# PROMPT OPTIMIZER — Always-on, uses existing groq_client
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
    """Optimize a prompt via Groq. Fast (~1 sec), near-zero cost."""
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

        # Strip markdown fences if present
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
    except Exception as e:
        return {
            "optimized_prompt": user_prompt,
            "changes_summary": "Optimizer unavailable — using original",
            "complexity": "medium",
            "task_type": "other",
            "recommended_model_tier": "mid",
        }


def render_optimization_card(result):
    """Show a compact card with optimization results."""
    tier = result.get("model_tier", "mid")
    tier_colors = {"budget": "#10b981", "mid": "#f59e0b", "premium": "#8b5cf6"}
    color = tier_colors.get(tier, "#94a3b8")

    savings_pct = result.get("savings_pct", 0)
    savings_display = f"-{savings_pct:.0f}% cost" if savings_pct > 1 else "optimal"

    changed = result.get("changed", False)
    if changed:
        change_line = (
            f'<div style="color:#8b949e; font-size:0.78rem; margin-top:0.3rem;">'
            f'✏️ {result.get("changes_summary", "")}'
            f'</div>'
        )
        optimized_line = (
            f'<details style="margin-top:0.4rem;">'
            f'<summary style="color:#58a6ff; font-size:0.78rem; cursor:pointer;">View optimized prompt</summary>'
            f'<div style="background:#0d1117; padding:0.5rem; border-radius:6px; margin-top:0.3rem; '
            f'font-size:0.78rem; color:#c9d1d9; white-space:pre-wrap; font-family:monospace;">'
            f'{result.get("optimized", "")}'
            f'</div>'
            f'</details>'
        )
    else:
        change_line = (
            f'<div style="color:#8b949e; font-size:0.78rem; margin-top:0.3rem;">'
            f'✓ Prompt already clean — no changes needed'
            f'</div>'
        )
        optimized_line = ""

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin: 0.3rem 0 0.6rem 0;
            font-size: 0.82rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{color}; font-weight:700; font-size:0.73rem; text-transform:uppercase; letter-spacing:0.04em;">
                    ⚡ Optimized → {result.get("model_label", "")}
                </span>
                <span style="color:#10b981; font-weight:600; font-size:0.78rem;">
                    {savings_display}
                </span>
            </div>
            {change_line}
            {optimized_line}
        </div>""",
        unsafe_allow_html=True,
    )


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

# Routing logic - all open source via Groq, complex via Claude
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
        return "llama-3.1-8b-instant", "Groq Llama 3 8B (Open source & fast!)", 0.00005, "groq"

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

        with st.spinner("⚡ Optimizing prompt..."):
            # ──────────────────────────────────────────
            # STEP 1: Optimize the prompt (always-on)
            # ──────────────────────────────────────────
            opt_result = optimize_prompt(query)
            optimized_query = opt_result.get("optimized_prompt", query)
            changed = query.strip().lower() != optimized_query.strip().lower()

            # Calculate cost comparison (before = Claude Opus baseline)
            baseline_cost_per_1m = 15.0  # Claude Opus as the "dumb default"
            before_cost = est_cost(query, baseline_cost_per_1m)

        with st.spinner("Routing and processing..."):
            try:
                # Build full query with file contents using OPTIMIZED prompt
                full_query = optimized_query
                if file_contents:
                    full_query = "\n\n".join(file_contents) + "\n\nUser question: " + optimized_query

                if mode == "Auto (Recommended)":
                    # Route based on the OPTIMIZED prompt
                    model_id, model_name, cost_per_1m, provider = route_query(
                        optimized_query,
                        has_files=len(file_contents) > 0,
                        has_images=len(image_data) > 0
                    )

                    # Calculate savings
                    after_cost = est_cost(optimized_query, cost_per_1m)
                    savings_pct = ((before_cost - after_cost) / before_cost * 100) if before_cost > 0 else 0

                    # Build optimization card data
                    opt_card = {
                        "model_tier": opt_result.get("recommended_model_tier", "mid"),
                        "model_label": model_name,
                        "savings_pct": savings_pct,
                        "changed": changed,
                        "changes_summary": opt_result.get("changes_summary", ""),
                        "optimized": optimized_query,
                    }

                    # Show the optimization card
                    render_optimization_card(opt_card)

                    # Update cumulative savings
                    st.session_state.total_savings += (before_cost - after_cost)
                    st.session_state.total_queries_optimized += 1
                    st.session_state.total_before_cost += before_cost

                    st.info("🎯 Routed to: **" + model_name + "**")

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
                            st.write("• Prompt optimization (via Llama 3.3 70B)")
                            st.write("• Keyword analysis")
                            st.write("• Length-based heuristics")
                            st.write("• File type detection")
                            st.write("• Cost-performance optimization")

                    # Prompt optimization details in observability
                    with st.expander("⚡ Prompt Optimization Details", expanded=False):
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
                    # ── MANUAL OVERRIDE MODE ──
                    # Still optimize the prompt, just don't override model choice
                    opt_card = {
                        "model_tier": opt_result.get("recommended_model_tier", "mid"),
                        "model_label": model_choice,
                        "savings_pct": 0,
                        "changed": changed,
                        "changes_summary": opt_result.get("changes_summary", ""),
                        "optimized": optimized_query,
                    }
                    render_optimization_card(opt_card)

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
    - **⚡ Automatic Prompt Optimization**: Every query is refined for clarity and efficiency before sending
    - Simple queries: Llama 3.1 8B via Groq (open source, Meta)
    - Medium queries: Llama 3.3 70B via Groq (open source, fast)
    - Complex reasoning/coding: Claude Sonnet 4
    - Image analysis: Claude Sonnet 4 (vision capabilities)

    Prompt Optimizer:
    - Removes filler words and redundancy
    - Adds specificity for better responses
    - Recommends the cheapest model that can handle the task
    - Runs via Groq Llama 3.3 70B (~1 sec, near-zero cost)

    Rate Limits:
    - 20 queries per hour per user (resets hourly)
    - Helps keep the service free and sustainable

    File Upload Features:
    - PDFs: Extract and analyze text
    - Text/Word docs: Process content
    - CSV/Excel: Data analysis and insights
    - Images: Visual analysis (via Claude)

    Why Groq?
    - Lightning fast responses (often under 1 second)
    - Runs open source Meta Llama models
    - Very affordable: $0.59 per 1M tokens
    - Generous free tier: 6,000 requests/day
    - Privacy-focused US company

    Cost Savings:
    - Llama 3.1 8B via Groq: near FREE for simple queries
    - Groq: 88% cheaper than GPT-4
    - Can achieve 95-100% savings on simple queries vs GPT-4
    """)
