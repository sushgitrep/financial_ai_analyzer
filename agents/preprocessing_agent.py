"""
Pre processing
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import requests

# import fitz  # PyMuPDF
# import pdfplumber
import nltk
from typing import Dict, Any, Optional, List
from datetime import datetime
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_manager import DataManager
from processors.transcript_processor_factory import TranscriptProcessorFactory

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class EnhancedPreprocessingAgent:
    """Final polished preprocessing agent"""

    def __init__(self):
        self.data_manager = DataManager()
        self.config = self.data_manager.config
        self.banks_config = self.data_manager.banks_config
        self.pdf_config = self.config.get("pdf", {})

        # Get current bank
        self.current_bank = st.session_state.get("current_bank")
        self.transcript_processor = TranscriptProcessorFactory.create_processor(
            self.current_bank
        )

        # Setup NLTK stopwords
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            self.stop_words = set()

    def run(self):
        """Run final polished preprocessing agent"""
        st.subheader("📋 Document Preprocessing")

        if not self.current_bank:
            st.error("❌ No bank selected. Please go to Bank Selection tab first.")
            return

        # Display current bank info (simple)
        try:
            bank_info = self.data_manager.get_bank_info(self.current_bank)
            bank_name = bank_info["name"] if bank_info else self.current_bank
        except:
            bank_name = self.current_bank

        st.info(f"**Processing for:** {bank_name}")

        # Always show fresh processing options (no load previous document)
        st.markdown("### 📄 Process Document")
        st.info("💡 Document will be automatically saved after processing")

        # Document input options
        input_method = st.radio(
            "Choose input method:",
            options=["📄 Use Default PDF from Config", "📤 Upload PDF File"],
            index=0,
            horizontal=True,
            key="input_method_polished",
        )

        if input_method == "📄 Use Default PDF from Config":
            self._handle_config_pdf()
        else:
            self._handle_file_upload()

        # Simple document status if loaded (no action buttons)
        if "document_data" in st.session_state:
            st.markdown("---")
            self._display_simple_status()

    def _display_simple_status(self):
        """Display simple document status - NO BUTTONS"""
        st.markdown("### ✅ Document Processed & Saved")
        st.success(
            "📄 Document has been automatically processed and saved. Ready for analysis!"
        )

    def _handle_config_pdf(self):
        """Handle PDF from config"""
        bank_info = self.banks_config.get("banks", {}).get(self.current_bank, {})
        pdf_url = bank_info.get("default_pdf_url")

        if not pdf_url:
            st.warning("⚠️ No default PDF URL configured for this bank")
            return

        st.info(f"**Default PDF URL:** {pdf_url}")

        if st.button(
            "🚀 Process PDF from Config",
            type="primary",
            use_container_width=True,
            key="process_config_pdf_polished",
        ):
            self._process_pdf_from_url(pdf_url)

    def _handle_file_upload(self):
        """Handle file upload"""
        uploaded_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"], key="pdf_upload_polished"
        )

        if uploaded_file is not None:
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)

            st.info(
                f"**File:** {uploaded_file.name} ({file_size / 1024 / 1024:.1f} MB)"
            )

            if st.button(
                "🚀 Process Uploaded PDF",
                type="primary",
                use_container_width=True,
                key="process_upload_pdf_polished",
            ):
                self._process_uploaded_pdf(uploaded_file)

    def _process_pdf_from_url(self, pdf_url: str):
        """Process PDF from URL with automatic save"""
        try:
            with st.spinner("📥 Downloading and processing PDF..."):
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                result = self._process_pdf_file(tmp_file_path, source=pdf_url)
                os.unlink(tmp_file_path)

                if result:
                    st.success("✅ PDF processed and automatically saved!")
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    def _process_uploaded_pdf(self, uploaded_file):
        """Process uploaded PDF with automatic save"""
        try:
            with st.spinner("📄 Processing uploaded PDF..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                result = self._process_pdf_file(
                    tmp_file_path, source=uploaded_file.name
                )
                os.unlink(tmp_file_path)

                if result:
                    st.success("✅ PDF processed and automatically saved!")
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    def _process_pdf_file(self, file_path: str, source: str = "unknown") -> bool:
        """Process PDF file with AUTOMATIC SAVE"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📄 Extracting text...")
            progress_bar.progress(20)

            # Extract with PyMuPDF
            text_pymupdf = self.transcript_processor.read_pdf_with_pymupdf(file_path)
            progress_bar.progress(40)

            # Extract with pdfplumber
            text_pdfplumber = self.transcript_processor.read_pdf_with_pdfplumber(
                file_path
            )
            progress_bar.progress(60)

            # Choose best extraction
            if len(text_pymupdf) > len(text_pdfplumber):
                raw_text = text_pymupdf
                method = "PyMuPDF"
            else:
                raw_text = text_pdfplumber
                method = "pdfplumber"

            if not raw_text or len(raw_text.strip()) < 50:
                st.error("❌ Could not extract sufficient text")
                return False

            status_text.text("🧹 Processing text...")
            progress_bar.progress(80)

            processed_data = self._preprocess_text(raw_text, source, method)

            # Store in session
            st.session_state.document_data = processed_data

            # AUTOMATIC SAVE
            status_text.text("💾 Automatically saving...")
            progress_bar.progress(90)

            # Clear existing analysis results
            session_keys_to_clear = [
                "topic_results",
                "sentiment_results",
                "summary_results",
            ]
            for key in session_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            # Auto-save to persistent storage
            try:
                self.data_manager.save_analysis_results(
                    self.current_bank, "document_data", processed_data
                )
                logger.info(f"Document automatically saved for {self.current_bank}")
            except Exception as e:
                logger.warning(f"Auto-save warning: {e}")

            progress_bar.progress(100)
            status_text.text("✅ Processing complete & saved!")

            return True
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            return False

    def _preprocess_text(
        self, raw_text: str, source: str, method: str
    ) -> Dict[str, Any]:
        """Preprocess text"""
        try:
            # Safe cleaning
            text = self._safe_clean_text(raw_text)
            cleaned_text = self._simple_nltk_processing(text)
            self.transcript_processor = TranscriptProcessorFactory.create_processor(
                self.current_bank
            )
            text_sections = self.transcript_processor.split_text_into_sections(
                self.transcript_processor.preprocess_text(raw_text)
            )

            return {
                "text": raw_text,
                "cleaned_text": cleaned_text,
                "text_sections": text_sections,
                "total_words": len(raw_text.split()),
                "total_pages": max(1, raw_text.count("\f") + 1),
                "cleaned_word_count": len(cleaned_text.split()),
                "source": source,
                "processed_at": datetime.now().isoformat(),
                "bank_key": self.current_bank,
                "bank_name": self.current_bank,
                "preprocessing_stats": {"extraction_method": method},
            }
        except Exception as e:
            logger.error(f"Text preprocessing error: {e}")
            return {
                "text": raw_text,
                "cleaned_text": raw_text,
                "text_sections": [],
                "total_words": len(raw_text.split()),
                "total_pages": 1,
                "cleaned_word_count": len(raw_text.split()),
                "source": source,
                "processed_at": datetime.now().isoformat(),
                "bank_key": self.current_bank,
                "bank_name": self.current_bank,
            }

    def _safe_clean_text(self, text: str) -> str:
        """Safe text cleaning"""
        try:
            replacements = {
                chr(8220): '"',
                chr(8221): '"',
                chr(8216): "'",
                chr(8217): "'",
                chr(8212): "--",
                chr(8211): "-",
                "\f": " ",
                "\r": " ",
                "\n": " ",
                "\t": " ",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return " ".join(text.split())
        except:
            return " ".join(text.split())

    def _simple_nltk_processing(self, text: str) -> str:
        """Simple NLTK processing"""
        try:
            tokens = word_tokenize(text.lower())
            processed = []
            for token in tokens:
                if token.isalpha() and len(token) > 2 and token not in self.stop_words:
                    processed.append(token)
            return " ".join(processed)
        except:
            return text.lower()


def run_preprocessing_agent():
    """Entry point"""
    agent = EnhancedPreprocessingAgent()
    agent.run()
