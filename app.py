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

st.markdown("**AI Should Be Free!**")

# File processing functions
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

        with st.spinner("Routing and processing..."):
            try:
                full_query = query
                if file_contents:
                    full_query = "\n\n".join(file_contents) + "\n\nUser question: " + query

                if mode == "Auto (Recommended)":
                    model_id, model_name, cost_per_1m, provider = route_query(
                        query,
                        has_files=len(file_contents) > 0,
                        has_images=len(image_data) > 0
                    )

                    st.info("🎯 Routed to: **" + model_name + "**")

                    query_lower = query.lower()
                    word_count = len(query.split())
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
                        "llama3-8b-8192": 0.0003,
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
                            if model_id == "llama3-8b-8192":
                                st.write("✅ Llama 3 8B (selected) - Open source, fast!")
                                st.write("⚪ Llama 3.3 70B - Larger, unnecessary for simple queries")
                                st.write("⚪ Claude Sonnet 4 - Expensive, overkill")
                            elif provider == "groq":
                                st.write("✅ Llama 3.3 70B (selected) - Fast & cheap!")
                                st.write("⚪ Llama 3 8B - Smaller, less capable for medium tasks")
                                st.write("⚪ Claude Sonnet 4 - Higher cost, unnecessary")
                            else:
                                st.write("✅ Claude Sonnet 4 (selected)")
                                st.write("⚪ Llama 3.3 70B - Lower cost but less capable for complex reasoning")
                                st.write("⚪ Llama 3 8B - Too small for this task")
                            st.write("**Routing Algorithm:**")
                            st.write("• Keyword analysis")
                            st.write("• Length-based heuristics")
                            st.write("• File type detection")
                            st.write("• Cost-performance optimization")

                else:
                    full_query_manual = query
                    if file_contents:
                        full_query_manual = "\n\n".join(file_contents) + "\n\nUser question: " + query

                    if "Llama 3 8B" in model_choice:
                        response = groq_client.chat.completions.create(
                            model="llama3-8b-8192",
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
    - Simple queries: Llama 3 8B via Groq (open source, Meta)
    - Medium queries: Llama 3.3 70B via Groq (open source, fast)
    - Complex reasoning/coding: Claude Sonnet 4
    - Image analysis: Claude Sonnet 4 (vision capabilities)

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
    - Llama 3 8B via Groq: near FREE for simple queries
    - Groq: 88% cheaper than GPT-4
    - Can achieve 95-100% savings on simple queries vs GPT-4
    """)
