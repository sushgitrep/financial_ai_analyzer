"""
Simple configuration for PDF Chatbot
Edit these settings to customize your chatbot
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Your OpenAI API Key (required)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI Model to use
OPENAI_MODEL = "gpt-4o-mini"  # Options: gpt-3.5-turbo, gpt-4, gpt-4-turbo
TEMPERATURE = 0.0  # 0.0 = focused, 1.0 = creative


# =============================================================================
# PDF CONFIGURATION  
# =============================================================================
# Path to your PDF file (if you don't want to upload via web interface)
# Leave empty to use file upload in Streamlit
DEFAULT_PDF_PATH = ""  # Example: "documents/my_document.pdf"


# =============================================================================
# VECTOR DATABASE CONFIGURATION
# =============================================================================
# Choose vector database: "faiss" or "chroma"
VECTOR_DB = "faiss"  # Options: "faiss" or "chroma"

# Document processing settings
CHUNK_SIZE = 1000        # Size of text chunks for processing
CHUNK_OVERLAP = 200      # Overlap between chunks
SIMILARITY_SEARCH_K = 5  # Number of similar documents to retrieve


# =============================================================================
# CHAT HISTORY CONFIGURATION
# =============================================================================
# How many previous messages to remember (user + assistant pairs)
# Set to 0 to disable chat history, higher numbers use more tokens
MAX_CHAT_HISTORY = 5  # Remember last 5 exchanges (10 messages total)

# Maximum tokens for responses
MAX_TOKENS = 1000


# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    """Check if configuration is valid"""
    errors = []

    if not os.getenv("OPENAI_API_KEY", ""):
        errors.append("❌ OPENAI_API_KEY is required")

    if VECTOR_DB not in ["faiss", "chroma"]:
        errors.append("❌ VECTOR_DB must be 'faiss' or 'chroma'")

    if MAX_CHAT_HISTORY < 0:
        errors.append("❌ MAX_CHAT_HISTORY must be >= 0")

    if CHUNK_SIZE < 100:
        errors.append("❌ CHUNK_SIZE too small (minimum 100)")

    return errors

