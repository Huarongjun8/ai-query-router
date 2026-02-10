import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient
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
hf_client = InferenceClient(token=st.secrets["HUGGINGFACE_API_KEY"])
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize rate limiting in session state
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
    st.session_state.reset_time = datetime.now() + timedelta(hours=1)

st.markdown("**AI Should Be Free!**")

# File processing functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(uploaded_file):
    """Extract text from Word document"""
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def process_csv_excel(uploaded_file, file_type):
    """Process CSV or Excel file"""
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Create summary
        summary = f"Dataset with {len(df)} rows and {len(df.columns)} columns.\n\n"
        summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary += f"First few rows:\n{df.head().to_string()}\n\n"
        summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
        summary += f"Basic statistics:\n{df.describe().to_string()}"
        return summary
    except Exception as e:
        return f"Error reading {file_type.upper()}: {str(e)}"

def image_to_base64(image_file):
    """Convert image to base64 for Claude API"""
    try:
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format=image.format if image.format else "PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Determine media type
        media_type = "image/png"
        if image.format == "JPEG":
            media_type = "image/jpeg"
        elif image.format == "WEBP":
            media_type = "image/webp"
        elif image.format == "GIF":
            media_type = "image/gif"
        
        return img_str, media_type
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# COMBINED INPUT AREA - Query + File Upload
query = st.text_area(
    "💬 Enter your query:", 
    height=100, 
    placeholder="Ask anything or ask about your uploaded files..."
)

# File uploader directly below (no scrolling needed on mobile)
uploaded_files = st.file_uploader(
    "📎 Attach files (optional)",
    type=["pdf", "txt", "docx", "csv", "xlsx", "png", "jpg", "jpeg", "webp", "gif"],
    accept_multiple_files=True,
    help="Drag and drop or click to upload PDFs, documents, spreadsheets, or images",
    label_visibility="visible"
)

# Show file count if files are attached (compact display)
if uploaded_files:
    st.caption(f"✅ {len(uploaded_files)} file(s) attached")

# Display rate limit counter in sidebar
st.sidebar.write(f"🔄 Queries used: {st.session_state.query_count}/20 this hour")
st.sidebar.caption(f"Resets at: {st.session_state.reset_time.strftime('%I:%M %p')}")

# Process uploaded files
file_contents = []
image_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == "pdf":
            content = extract_text_from_pdf(uploaded_file)
            file_contents.append(f"[PDF: {uploaded_file.name}]\n{content}\n")
            
        elif file_type == "txt":
            content = uploaded_file.read().decode('utf-8')
            file_contents.append(f"[Text file: {uploaded_file.name}]\n{content}\n")
            
        elif file_type == "docx":
            content = extract_text_from_docx(uploaded_file)
            file_contents.append(f"[Word document: {uploaded_file.name}]\n{content}\n")
            
        elif file_type in ["csv", "xlsx"]:
            content = process_csv_excel(uploaded_file, file_type)
            file_contents.append(f"[Spreadsheet: {uploaded_file.name}]\n{content}\n")
            
        elif file_type in ["png", "jpg", "jpeg", "webp", "gif"]:
            img_base64, media_type = image_to_base64(uploaded_file)
            if img_base64:
                image_data.append({
                    "name": uploaded_file.name,
                    "base64": img_base64,
                    "media_type": media_type
                })
                # Show image preview in an expander to keep UI clean
                with st.expander(f"🖼️ Preview: {uploaded_file.name}"):
                    st.image(uploaded_file, width=300)
            else:
                st.error(media_type)  # Error message

# Routing logic
def route_query(query, has_files=False, has_images=False):
    """Determine best model based on query complexity and file types"""
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Force Claude if images are present (only Claude has vision)
    if has_images:
        return "claude-sonnet-4-20250514", "Claude Sonnet 4 (Image analysis)", 0.003, "anthropic"
    
    # Complex reasoning indicators
    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy', 'summarize']
    complexity_score = sum(1 for keyword in complex_keywords if keyword in query_lower)
    
    # If file uploaded, add complexity
    if has_files:
        complexity_score += 1
    
    # Route decision with Groq for medium complexity
    if complexity_score >= 2 or word_count > 50:
        return "claude-sonnet-4-20250514", "Claude Sonnet 4 (Complex reasoning)", 0.003, "anthropic"
    elif word_count > 20 or complexity_score == 1:
        return "llama-3.3-70b-versatile", "Groq Llama 3.3 70B (Fast & cheap!)", 0.00059, "groq"
    else:
        return "Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B (Simple query - Open source & free!)", 0.00000, "huggingface"

# Mode selection
mode = st.radio("Routing mode:", ["Auto (Recommended)", "Manual Override"])

if mode == "Manual Override":
    model_choice = st.selectbox("Choose model:", 
        ["Qwen 2.5 (Open Source - Free)", "Groq Llama 3.3 - Fast & Cheap", "GPT-4o Mini", "Claude Sonnet 4"])

# Submit button (full width for mobile)
if st.button("🚀 Send Query", type="primary", use_container_width=True):
    if query or uploaded_files:
        # CHECK RATE LIMIT BEFORE API CALLS
        if st.session_state.query_count >= 20:
            if datetime.now() < st.session_state.reset_time:
                st.error(f"⏱️ Rate limit reached! You've used {st.session_state.query_count}/20 queries this hour.")
                st.info(f"Your limit resets at {st.session_state.reset_time.strftime('%I:%M %p')}. Please try again then!")
                st.caption("Rate limits help us keep the service free and sustainable. Thank you for understanding! 🙏")
                st.stop()
            else:
                # Reset counter if time window has passed
                st.session_state.query_count = 0
                st.session_state.reset_time = datetime.now() + timedelta(hours=1)
        
        with st.spinner("Routing and processing..."):
            try:
                # Combine file contents with query
                full_query = query
                if file_contents:
                    full_query = "\n\n".join(file_contents) + "\n\nUser question: " + query
                
                # Auto routing
                if mode == "Auto (Recommended)":
                    model_id, model_name, cost_per_1m, provider = route_query(
                        query, 
                        has_files=len(file_contents) > 0,
                        has_images=len(image_data) > 0
                    )
                    
                    # Routing explainability
                    st.info(f"🎯 Routed to: **{model_name}**")

                    # Calculate routing confidence and alternatives
                    query_lower = query.lower()
                    word_count = len(query.split())
                    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy', 'summarize']
                    complexity_score = sum(1 for keyword in complex_keywords if keyword in query_lower)

                    # Determine confidence
                    if len(image_data) > 0:
                        confidence = 100
                        reasoning = f"Image analysis requires vision model (only Claude supports this)"
                    elif complexity_score >= 2 or word_count > 50:
                        confidence = 90 + min(complexity_score * 2, 10)
                        reasoning = f"High complexity detected ({complexity_score} complex keywords, {word_count} words)"
                    elif word_count < 10 and complexity_score == 0:
                        confidence = 95
                        reasoning = f"Simple query ({word_count} words, no complex keywords) - using free open-source Qwen"
                    else:
                        confidence = 70 + (complexity_score * 5)
                        reasoning = f"Medium complexity ({complexity_score} complex keywords, {word_count} words)"
                    
                    if file_contents:
                        reasoning += f", {len(file_contents)} file(s) uploaded"
                    
                    # Make API call based on provider
                    if provider == "huggingface":
                        messages = [{"role": "user", "content": full_query}]
                        response = hf_client.chat_completion(
                            messages=messages,
                            model=model_id,
                            max_tokens=1000
                        )
                        answer = response.choices[0].message.content
                        # Estimate tokens for HF
                        tokens_used = len(full_query.split()) * 1.3 + len(answer.split()) * 1.3
                        
                    elif provider == "groq":
                        response = groq_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": full_query}],
                            temperature=0.3,
                            max_tokens=2048
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens
                        
                    elif provider == "anthropic":
                        # Handle images for Claude
                        if image_data:
                            content_parts = []
                            
                            # Add images first
                            for img in image_data:
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": img["media_type"],
                                        "data": img["base64"]
                                    }
                                })
                            
                            # Add text content
                            content_parts.append({
                                "type": "text",
                                "text": full_query
                            })
                            
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
                        
                    else:  # openai
                        response = openai_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": full_query}]
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens
                    
                    # INCREMENT COUNTER AFTER SUCCESSFUL API CALL
                    st.session_state.query_count += 1
                    
                    # Display response FIRST
                    st.success("Response:")
                    st.write(answer)
                    
                    # Then show metrics below
                    # Cost calculation
                    estimated_cost = (tokens_used / 1000000) * cost_per_1m
                    
                    # Calculate what GPT-4 would have cost (baseline comparison)
                    gpt4_cost_per_1m = 0.005  # $5 per 1M input tokens
                    gpt4_would_cost = (tokens_used / 1000000) * gpt4_cost_per_1m
                    
                    # Calculate savings
                    savings = gpt4_would_cost - estimated_cost
                    savings_percent = (savings / gpt4_would_cost) * 100 if gpt4_would_cost > 0 else 0
                    
                    # Environmental impact estimates (Wh per 1000 tokens)
                    energy_rates = {
                        "gpt-4o-mini": 0.001,
                        "gpt-4o": 0.003,
                        "claude-sonnet-4-20250514": 0.002,
                        "Qwen/Qwen2.5-72B-Instruct": 0.0005,
                        "llama-3.3-70b-versatile": 0.0008
                    }
                    
                    # Calculate energy used
                    energy_per_1k = energy_rates.get(model_id, 0.002)
                    energy_used_wh = (tokens_used / 1000) * energy_per_1k
                    
                    # GPT-4 baseline energy
                    gpt4_energy_wh = (tokens_used / 1000) * 0.003
                    energy_saved_wh = gpt4_energy_wh - energy_used_wh
                    
                    # CO2 estimate (0.4 kg CO2 per kWh average grid)
                    co2_grams = (energy_used_wh / 1000) * 400
                    co2_saved_grams = (energy_saved_wh / 1000) * 400
                    
                    # Display cost metrics
                    st.subheader("💵 Cost Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Smart Routing Cost", f"${estimated_cost:.6f}")
                    with col2:
                        st.metric("GPT-4 Would Cost", f"${gpt4_would_cost:.6f}")
                    with col3:
                        st.metric("💰 You Saved", f"${savings:.6f}", delta=f"-{savings_percent:.1f}%", delta_color="inverse")
                    
                    # Display environmental metrics
                    st.subheader("🌍 Environmental Impact")
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        st.metric("Energy Used", f"{energy_used_wh:.4f} Wh")
                    with col5:
                        st.metric("CO2 Emissions", f"{co2_grams:.3f} g")
                    with col6:
                        st.metric("🌱 Energy Saved", f"{energy_saved_wh:.4f} Wh", delta=f"{co2_saved_grams:.3f}g CO2", delta_color="inverse")
                    
                    st.caption("⚠️ Environmental estimates are approximate based on published AI energy research")
                    
                    # Observability panel (collapsed by default)
                    with st.expander("🔍 Routing Observability & Explainability", expanded=False):
                        obs_col1, obs_col2 = st.columns(2)
                        
                        with obs_col1:
                            st.metric("Routing Confidence", f"{confidence}%")
                            st.write("**Decision Reasoning:**")
                            st.write(f"• {reasoning}")
                            st.write(f"• Query length: {word_count} words")
                            st.write(f"• Complexity indicators: {complexity_score}")
                            if file_contents:
                                st.write(f"• Files analyzed: {len(file_contents)}")
                            if image_data:
                                st.write(f"• Images analyzed: {len(image_data)}")
                        
                        with obs_col2:
                            st.write("**Alternative Models Considered:**")
                            if provider == "huggingface":
                                st.write("✅ Qwen 2.5 72B (selected) - Open source, FREE!")
                                st.write("⚪ Groq Llama 3.3 - Fast but unnecessary")
                                st.write("⚪ Claude Sonnet 4 - Expensive, overkill")
                            elif provider == "groq":
                                st.write("✅ Groq Llama 3.3 (selected) - Fast & cheap!")
                                st.write("⚪ Qwen 2.5 - Free but less capable for medium tasks")
                                st.write("⚪ Claude Sonnet 4 - Higher cost, unnecessary")
                            else:
                                st.write("✅ Claude Sonnet 4 (selected)")
                                st.write("⚪ Groq Llama 3.3 - Lower cost but less capable for complex reasoning")
                                st.write("⚪ Qwen 2.5 - Free but insufficient for this task")
                            
                            st.write("**Routing Algorithm:**")
                            st.write("• Keyword analysis")
                            st.write("• Length-based heuristics")
                            st.write("• File type detection")
                            st.write("• Cost-performance optimization")
                    
                # Manual override
                else:
                    # Prepare content for manual mode
                    full_query_manual = query
                    if file_contents:
                        full_query_manual = "\n\n".join(file_contents) + "\n\nUser question: " + query
                    
                    if "Qwen" in model_choice:
                        messages = [{"role": "user", "content": full_query_manual}]
                        response = hf_client.chat_completion(
                            messages=messages,
                            model="Qwen/Qwen2.5-72B-Instruct",
                            max_tokens=1000
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
                        # Claude with image support
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
                    
                    # INCREMENT COUNTER AFTER SUCCESSFUL API CALL (Manual mode too)
                    st.session_state.query_count += 1
                    
                    # Display response for manual mode
                    st.success("Response:")
                    st.write(answer)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                # Don't increment counter if there was an error
    else:
        st.warning("Please enter a query or upload files!")

# Info section
with st.expander("ℹ️ How Auto-Routing Works"):
    st.write("""
    **Smart routing saves you 70-100% on AI costs AND reduces environmental impact by:**
    - Ultra-simple queries → Qwen 2.5 72B (open source, FREE!)
    - Simple/medium queries → Groq Llama 3.3 (⚡ super fast, cheap)
    - Complex reasoning/coding → Claude Sonnet 4
    - Image analysis → Claude Sonnet 4 (vision capabilities)
    
    **Rate Limits:**
    - 20 queries per hour per user (resets hourly)
    - Helps keep the service free and sustainable
    
    **File Upload Features:**
    - 📄 PDFs - Extract and analyze text
    - 📝 Text/Word docs - Process content
    - 📊 CSV/Excel - Data analysis and insights
    - 🖼️ Images - Visual analysis (via Claude)
    
    **Complexity indicators:**
    - Keywords: analyze, explain, compare, code, debug, etc.
    - Query length > 50 words
    - Technical or strategic questions
    - File uploads automatically increase complexity
    
    **Why Groq?**
    - ⚡ Lightning fast responses (often <1 second!)
    - 💰 Very affordable: $0.59 per 1M tokens
    - 🎁 Generous free tier: 6,000 requests/day
    - 🔒 Privacy-focused US company
    
    **Environmental Impact:**
    - Energy estimates based on model size and token usage
    - CO2 calculations use average grid emissions (0.4 kg/kWh)
    - Smart routing typically saves 40-85% energy vs. always using GPT-4
    - Open source models like Qwen are often more energy efficient
    
    **Cost Savings:**
    - Qwen 2.5: FREE (open source via Hugging Face!)
    - Groq: 88% cheaper than GPT-4 ($0.59 vs $5 per 1M tokens)
    - Can achieve 95-100% savings on simple queries vs GPT-4
    """)
