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
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

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
