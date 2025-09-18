"""
Financial AI Analyzer v3.0 - FINAL POLISHED VERSION
Clean header, fixed bank selection, no feature sections
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import utilities
from utils.data_manager import DataManager
from agents.chatbot_agent import initialize_chatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import subprocess
import streamlit as st

@st.cache_resource
def download_spacy_model():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Call this function at the start of your app
download_spacy_model()


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Financial AI Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def display_main_header():

    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem 1rem; margin-bottom: 2rem; border-radius: 15px;">
        <h1>🏦 Financial AI Analyzer</h1>
    </div>
    """, unsafe_allow_html=True)

def display_bank_selection_tab():
    """Display polished bank selection tab with FIXED selection logic"""
    st.markdown("### 🏪 Bank Selection")

    # Initialize data manager
    try:
        data_manager = DataManager()
        banks = data_manager.get_bank_list()
    except Exception as e:
        st.error(f"❌ Error loading banks: {str(e)}")
        return

    if not banks:
        st.error("❌ No banks configuration found")
        return

    # Get current bank
    current_bank = st.session_state.get('current_bank', None)

    # Handle confirmation state
    if 'confirm_bank_switch' in st.session_state:
        _handle_switch_confirmation(st.session_state.confirm_bank_switch, data_manager)
        return

    # Clean dropdown selection
    st.markdown("#### 🏦 Select Bank for Analysis")

    # Create options for dropdown
    bank_options = ["-- Select a Bank --"] + [f"{bank['name']} ({bank['ticker']})" for bank in banks]
    bank_keys = [None] + [bank['key'] for bank in banks]

    # Find current selection index
    current_index = 0
    if current_bank:
        try:
            current_index = bank_keys.index(current_bank)
        except ValueError:
            current_index = 0

    # Proper alignment 
    col1, col2 = st.columns([4, 1])

    with col1:
        selected_index = st.selectbox(
            "Choose your bank:",
            range(len(bank_options)),
            format_func=lambda x: bank_options[x],
            index=current_index,
            key="bank_selector_polished"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Alignment spacing
        if selected_index > 0:  # Valid bank selected
            selected_bank_key = bank_keys[selected_index]

            # FIXED: Immediate state change detection and handling
            if selected_bank_key != current_bank:
                if st.button("🔄 Select Bank", type="primary", use_container_width=True, key="select_bank_polished"):
                    _initiate_bank_selection(selected_bank_key, data_manager)
            else:
                st.success("✅ Selected")
        else:
            st.button("Select Bank", disabled=True, use_container_width=True, key="select_bank_disabled_polished")

    # Show current selection only
    if current_bank:
        try:
            bank_info = data_manager.get_bank_info(current_bank)
            bank_name = bank_info['name'] if bank_info else current_bank
            st.info(f"**Currently Selected:** {bank_name}")
        except Exception as e:
            st.error(f"Error displaying bank info: {str(e)}")

def _initiate_bank_selection(new_bank_key: str, data_manager: DataManager):
    """FIXED: Initiate bank selection with proper state handling"""
    current_bank = st.session_state.get('current_bank')

    # If no current bank, select directly
    if not current_bank:
        success = _perform_bank_selection(new_bank_key, data_manager)
        if success:
            st.success(f"✅ **Bank Selected!**")
            st.rerun()
        return

    # If same bank, nothing to do
    if current_bank == new_bank_key:
        st.info("This bank is already selected!")
        return

    # Check if current bank has data - FIXED logic
    try:
        current_bank_info = data_manager.get_bank_info(current_bank)
        has_current_data = current_bank_info['has_data'] if current_bank_info else False

        if has_current_data:
            # Store confirmation state - FIXED to work on first click
            st.session_state.confirm_bank_switch = {
                'current_bank': current_bank,
                'new_bank': new_bank_key,
                'current_name': current_bank_info['name'] if current_bank_info else current_bank,
                'new_name': data_manager.get_bank_info(new_bank_key)['name'] if data_manager.get_bank_info(new_bank_key) else new_bank_key
            }
            st.rerun()  # Force immediate rerun to show confirmation
        else:
            # Switch directly if no data
            success = _perform_bank_selection(new_bank_key, data_manager)
            if success:
                st.success(f"✅ **Bank Selected!**")
                st.rerun()

    except Exception as e:
        st.error(f"Error checking bank data: {str(e)}")

def _handle_switch_confirmation(switch_info: dict, data_manager: DataManager):
    """Handle bank switch confirmation dialog"""

    st.warning(f"""
    ⚠️ **Bank Switch Warning**

    You have existing data for **{switch_info['current_name']}**.  
    Switching to **{switch_info['new_name']}** will reset the current session 
    (but saved data will be preserved).
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, Switch Bank", type="primary", use_container_width=True, key="confirm_switch_polished"):
            success = _perform_bank_selection(switch_info['new_bank'], data_manager)
            if success:
                del st.session_state.confirm_bank_switch
                st.success(f"✅ **Switched to {switch_info['new_name']}!**")
                st.rerun()

    with col2:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_switch_polished"):
            del st.session_state.confirm_bank_switch
            st.info("Bank switch cancelled.")
            st.rerun()

def _perform_bank_selection(new_bank_key: str, data_manager: DataManager):
    """Perform actual bank selection"""
    try:
        # Clear current session data
        session_keys_to_clear = [
            'document_data', 'topic_results', 'sentiment_results', 'summary_results', 'chat_history', 'raw_text', 'pdf_processed', 'filepath'
        ]

        for key in session_keys_to_clear:
            if key in st.session_state:
                logger.info(str(key))
                del st.session_state[key]
        
        for key in st.session_state:
           logger.info("session key : " + str(key))
           
        # Check if chatbot exists in session state
        if "chatbot" in st.session_state:
            logger.info("history chat ")
            # st.session_state.chatbot, error = initialize_chatbot()
            st.session_state.chatbot.clear_chat_history()
            
            # if error:
            #     st.error(f"Chatbot failed to initialize: {error}")
            # elif st.session_state.chatbot is None:
            #     st.error("Chatbot object is None for unknown reason")
            # st.session_state.chatbot.process_pdf(st.session_state.filepath)
        if "messages" in st.session_state:
            logger.info("messages")
            st.session_state.messages = []

        # Set new bank
        st.session_state.current_bank = new_bank_key

        logger.info(f"Bank selected: {new_bank_key}")
        return True

    except Exception as e:
        st.error(f"Error selecting bank: {str(e)}")
        logger.error(f"Bank selection error: {e}")
        return False

def main():
    configure_page()
    display_main_header()
    download_spacy_model()

    # REMOVED: New features section completely removed

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏪 Bank Selection",
        "📋 Preprocessing", 
        "🎯 Topic Modeling",
        "💭 Sentiment Analysis",
        "📝 Summarization",
        "🤖 Chatbot"
    ])

    # Tab 1: Bank Selection
    with tab1:
        display_bank_selection_tab()

    # Other tabs - require bank selection
    current_bank = st.session_state.get('current_bank')

    if not current_bank:
        # Show placeholder
        for tab_obj in [tab2, tab3, tab4, tab5, tab6]:
            with tab_obj:
                st.info("👈 Please select a bank from the Bank Selection tab first")
    else:
        # Import and run agents
        try:
            from agents.preprocessing_agent import run_preprocessing_agent
            from agents.topic_modeling_agent import run_topic_modeling_agent  
            from agents.sentiment_analysis_agent import run_sentiment_analysis_agent
            from agents.summarization_agent import run_summarization_agent
            from agents.chatbot_agent import run_chatbot_agent

            with tab2:
                try:
                    run_preprocessing_agent()
                except Exception as e:
                    st.error(f"❌ Preprocessing Error: {str(e)}")

            with tab3:
                try:
                    run_topic_modeling_agent()
                except Exception as e:
                    st.error(f"❌ Topic Modeling Error: {str(e)}")

            with tab4:
                try:
                    run_sentiment_analysis_agent()
                except Exception as e:
                    st.error(f"❌ Sentiment Analysis Error: {str(e)}")

            with tab5:
                try:
                    run_summarization_agent()
                except Exception as e:
                    st.error(f"❌ Summarization Error: {str(e)}")

            with tab6:
                try:
                    run_chatbot_agent()
                except Exception as e:
                    st.error(f"❌ Chatbot Error: {str(e)}")

        except ImportError as e:
            st.error(f"❌ Agent Import Error: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info(f"Main application main")
        main()
    except Exception as e:
        st.error(f"💥 Application Error: {str(e)}")
        logger.error(f"Main application error: {e}")