import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient
from groq import Groq

# Initialize API clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
hf_client = InferenceClient(token=st.secrets["HUGGINGFACE_API_KEY"])
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("🤖 AI Query Router - Auto Mode")
st.write("Automatically routes your query to the most cost-effective model")

# Routing logic
def route_query(query):
    """Determine best model based on query complexity"""
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Complex reasoning indicators
    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy']
    complexity_score = sum(1 for keyword in query_lower if keyword in complex_keywords)
    
    # Route decision with Groq for medium complexity
    if complexity_score >= 2 or word_count > 50:
        return "claude-sonnet-4-20250514", "Claude Sonnet 4 (Complex reasoning)", 0.003, "anthropic"
    elif word_count > 20 or complexity_score == 1:
        return "llama-3.3-70b-versatile", "Groq Llama 3.3 70B (Fast & cheap!)", 0.00059, "groq"
    else:
        return "Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B (Simple query - Open source & free!)", 0.00000, "huggingface"
    
# User input
query = st.text_area("Enter your query:", height=100, placeholder="Ask anything...")

# Mode selection
mode = st.radio("Routing mode:", ["Auto (Recommended)", "Manual Override"])

if mode == "Manual Override":
    model_choice = st.selectbox("Choose model:", 
        ["Qwen 2.5 (Open Source - Free)", "Groq Llama 3.3 - Fast & Cheap", "GPT-4o Mini", "Claude Sonnet 4"])

if st.button("Send Query", type="primary"):
    if query:
        with st.spinner("Routing and processing..."):
            try:
                # Auto routing
                if mode == "Auto (Recommended)":
                    model_id, model_name, cost_per_1m, provider = route_query(query)
                    
                    # Routing explainability
                    st.info(f"🎯 Routed to: **{model_name}**")

                    # Calculate routing confidence and alternatives
                    query_lower = query.lower()
                    word_count = len(query.split())
                    complex_keywords = ['analyze', 'explain', 'compare', 'evaluate', 'design', 'create', 'code', 'debug', 'strategy']
                    complexity_score = sum(1 for keyword in query_lower if keyword in complex_keywords)

                    # Determine confidence
                    if complexity_score >= 2 or word_count > 50:
                        confidence = 90 + min(complexity_score * 2, 10)
                        reasoning = f"High complexity detected ({complexity_score} complex keywords, {word_count} words)"
                    elif word_count < 10 and complexity_score == 0:
                        confidence = 95
                        reasoning = f"Simple query ({word_count} words, no complex keywords) - using free open-source Qwen"
                    else:
                        confidence = 70 + (complexity_score * 5)
                        reasoning = f"Medium complexity ({complexity_score} complex keywords, {word_count} words)"
                    
                    # Make API call based on provider
                    if provider == "huggingface":
                        messages = [{"role": "user", "content": query}]
                        response = hf_client.chat_completion(
                            messages=messages,
                            model=model_id,
                            max_tokens=500
                        )
                        answer = response.choices[0].message.content
                        # Estimate tokens for HF
                        tokens_used = len(query.split()) * 1.3 + len(answer.split()) * 1.3
                        
                    elif provider == "groq":
                        response = groq_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": query}],
                            temperature=0.3,
                            max_tokens=1024
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens
                        
                    elif provider == "anthropic":
                        response = anthropic_client.messages.create(
                            model=model_id,
                            max_tokens=1024,
                            messages=[{"role": "user", "content": query}]
                        )
                        answer = response.content[0].text
                        tokens_used = response.usage.input_tokens + response.usage.output_tokens
                        
                    else:  # openai
                        response = openai_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": query}]
                        )
                        answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens
                    
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
                            st.write("• Cost-performance optimization")
                    
                # Manual override
                else:
                    if "Qwen" in model_choice:
                        messages = [{"role": "user", "content": query}]
                        response = hf_client.chat_completion(
                            messages=messages,
                            model="Qwen/Qwen2.5-72B-Instruct",
                            max_tokens=500
                        )
                        answer = response.choices[0].message.content
                    elif "Groq" in model_choice:
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": query}]
                        )
                        answer = response.choices[0].message.content
                    elif "GPT" in model_choice:
                        model = "gpt-4o-mini" if "Mini" in model_choice else "gpt-4o"
                        response = openai_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": query}]
                        )
                        answer = response.choices[0].message.content
                    else:
                        response = anthropic_client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            messages=[{"role": "user", "content": query}]
                        )
                        answer = response.content[0].text
                    
                    # Display response for manual mode
                    st.success("Response:")
                    st.write(answer)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query first!")

# Info section
with st.expander("ℹ️ How Auto-Routing Works"):
    st.write("""
    **Smart routing saves you 70-100% on AI costs AND reduces environmental impact by:**
    - Ultra-simple queries → Qwen 2.5 72B (open source, FREE!)
    - Simple/medium queries → Groq Llama 3.3 (⚡ super fast, cheap)
    - Complex reasoning/coding → Claude Sonnet 4
    
    **Complexity indicators:**
    - Keywords: analyze, explain, compare, code, debug, etc.
    - Query length > 50 words
    - Technical or strategic questions
    
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