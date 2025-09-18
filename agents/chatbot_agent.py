"""
Simple PDF Chatbot - Streamlit Web Interface
Single page application for PDF chat with memory
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from loguru import logger

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Simple PDF Chatbot",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Import our components
try:
    from utils.chat import SimplePDFChatbot
    import chatbot_config
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Make sure all files are in the same directory and dependencies are installed")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success { background-color: #d4edda; }
    .status-warning { background-color: #fff3cd; }
    .status-error { background-color: #f8d7da; }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message { 
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .bot-message { 
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)
from loguru import logger
@st.cache_resource
def initialize_chatbot():
    logger.info("ch agent initia")
    """Initialize chatbot (cached for performance)"""
    try:
        return SimplePDFChatbot(), None
    except Exception as e:
        return None, str(e)

def display_chat_history():
    """Display chat history in a nice format"""
    if 'chatbot' in st.session_state and st.session_state.chatbot:
        history = st.session_state.chatbot.get_chat_history()

        if history:
            st.subheader("💬 Chat History")
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong> {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Bot:</strong> {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("💭 No chat history yet. Start a conversation!")

def run_chatbot_agent():
    """Main Streamlit application"""
    
    if os.environ.get('OPENAI_API_KEY') == None:   # only proceed if user entered something
        api_key = st.chat_input("Enter API key")
        if api_key:  # only assign if user entered something
            os.environ['OPENAI_API_KEY'] = api_key
            st.success("API Key is set ✅" + str(os.getenv("OPENAI_API_KEY", "")))
            st.session_state.chatbot, error = initialize_chatbot()
            if error:
                st.error(f"Chatbot failed to initialize: {error}")
            elif st.session_state.chatbot is None:
                st.error("Chatbot object is None for unknown reason")
            # st.session_state.chatbot.process_pdf(st.session_state.filepath) 
    
    if 'chatbot' in st.session_state and 'raw_text' in st.session_state:
        logger.info("chatbot agent 112")
        if 'pdf_processed' not in st.session_state and os.environ.get('OPENAI_API_KEY') != None:
            logger.info("chatbot agent 114")
            st.session_state.chatbot.process_pdf(st.session_state.filepath)
    
    # Header
    st.markdown('<div class="main-header">🤖 Simple PDF Chatbot</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
        st.session_state.messages = []

        # Status and Controls
        st.header("CHATBOT")

        if not st.session_state.chatbot:
            st.markdown('<div class="status-box status-error">❌ Chatbot Not Initialized</div>', unsafe_allow_html=True)

    if 'pdf_processed' in st.session_state:
        # Chat Interface
        col1, = st.columns(1)

        with col1:
            st.header("💬 Chat with Your Document")

            # Chat messages container
            chat_container = st.container()

            # Display existing messages
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                        <strong>👤 You:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                        <strong>🤖 Bot:</strong> {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)

        # Chat input
        
            
            user_input = st.chat_input("Ask a question about your document...")
            logger.info("dddeee")
            if user_input:
                logger.info("eee")
                # Add user message to session
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Get bot response
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.chatbot.chat(user_input)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_response = f"❌ Error: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": error_response})

                # Rerun to show new messages
                st.rerun()


if __name__ == "__main__":
    run_chatbot_agent()