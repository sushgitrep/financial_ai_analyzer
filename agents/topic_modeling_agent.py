"""
Topic Modelling
"""

import streamlit as st
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from collections import Counter
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_manager import DataManager

logger = logging.getLogger(__name__)

class TopicModelingAgent:
    """Final polished topic modeling agent"""

    def __init__(self):
        self.data_manager = DataManager()
        self.current_bank = st.session_state.get('current_bank')

    def run(self):
        """Run polished topic modeling with correct order"""
        st.subheader("🎯 Topic Modeling with Visualizations")

        if not self.current_bank:
            st.error("❌ No bank selected. Please go to Bank Selection tab first.")
            return

        # Check prerequisites
        if 'document_data' not in st.session_state:
            st.warning("⚠️ Please process a document first in the Preprocessing tab")
            return

        bank_info = self.data_manager.get_bank_info(self.current_bank)
        st.info(f"**Analyzing:** {bank_info['name'] if bank_info else self.current_bank}")

        # 1. Analysis section FIRST
        st.markdown("### 🎯 Run Topic Analysis")

        available_plots = self.data_manager.get_available_plots(self.current_bank)
        if available_plots:
            st.warning("⚠️ Running new analysis will overwrite previous topic plots")

        if st.button("🎯 Run Topic Analysis", type="primary", use_container_width=True, key="run_topic_analysis_polished"):
            self._run_topic_analysis()

        # 2. Current Plots section SECOND (after running analysis)
        if 'topic_results' in st.session_state:
            st.markdown("---")
            self._display_current_plots()

        # 3. FIXED: Previous Analysis section THIRD (only one instance, no duplicates)
        if available_plots:
            st.markdown("---")
            self._show_previous_analysis_section(available_plots)

        # 4. Analysis results LAST
        if 'topic_results' in st.session_state:
            st.markdown("---")
            self._display_current_results()

    def _show_previous_analysis_section(self, available_plots: Dict):
        """Show previous analysis plots section - SINGLE INSTANCE ONLY"""
        st.markdown("### 📂 Previous Analysis Plots")

        if st.button("📊 View Previous Plots", type="secondary", use_container_width=True, key="view_topic_plots_polished"):
            self._show_previous_plots(available_plots)

    def _show_previous_plots(self, available_plots: Dict):
        """Display previous topic modeling plots"""
        topic_plots = {}
        for plot_name, plot_files in available_plots.items():
            if any(keyword in plot_name.lower() for keyword in ['topic', 'word', 'frequency', 'analysis', 'cloud']):
                topic_plots[plot_name] = plot_files

        if topic_plots:
            cols = st.columns(2)
            for i, (plot_name, plot_files) in enumerate(topic_plots.items()):
                with cols[i % 2]:
                    st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
                    if plot_files:
                        try:
                            latest_plot = plot_files[0]
                            st.image(latest_plot, caption=f"Previous: {plot_name}", use_container_width=True)
                        except Exception as e:
                            st.write(f"Plot file: {Path(latest_plot).name}")
        else:
            st.info("No previous topic analysis plots found.")

    def _run_topic_analysis(self):
        """Run actual topic analysis"""
        try:
            with st.spinner("🎯 Running topic analysis with word cloud..."):
                doc_data = st.session_state.document_data
                text = doc_data.get('cleaned_text', doc_data.get('text', ''))

                # Real topic analysis
                results = self._perform_real_topic_analysis(text)

                # Save results to session
                st.session_state.topic_results = results

                # Save to persistent storage (overwriting previous)
                self.data_manager.save_analysis_results(self.current_bank, 'topic_results', results)

                # Generate and save ONLY 2 plots
                self._generate_and_save_plots(results)

                st.success("🎯 Topic analysis completed and saved!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Topic analysis error: {e}")

    def _perform_real_topic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform real topic analysis"""

        # Financial topic keywords
        financial_topics = {
            'Credit Risk & Lending': ['credit', 'loan', 'default', 'provision', 'risk', 'lending', 'borrower', 'delinquency'],
            'Financial Performance': ['revenue', 'profit', 'earnings', 'income', 'growth', 'performance', 'results', 'margin'],
            'Market & Economic Outlook': ['market', 'outlook', 'forecast', 'guidance', 'future', 'economic', 'trends', 'conditions'],
            'Digital Banking & Technology': ['digital', 'technology', 'innovation', 'fintech', 'mobile', 'online', 'platform', 'automation'],
            'Regulatory & Compliance': ['regulation', 'regulatory', 'compliance', 'capital', 'basel', 'federal', 'policy', 'oversight'],
            'Investment Banking': ['investment', 'banking', 'securities', 'trading', 'underwriting', 'advisory', 'markets', 'equity'],
            'Consumer Banking': ['consumer', 'retail', 'deposits', 'checking', 'savings', 'mortgage', 'personal', 'customer'],
            'Risk Management': ['risk', 'management', 'operational', 'liquidity', 'stress', 'testing', 'mitigation', 'exposure']
        }

        # Analyze text for topics
        text_lower = text.lower()
        topics_data = []

        for topic_name, keywords in financial_topics.items():
            total_count = sum(text_lower.count(keyword) for keyword in keywords)

            if total_count > 0:
                relevance_score = min(total_count / 50.0, 1.0)

                topics_data.append({
                    'topic_id': len(topics_data),
                    'name': topic_name,
                    'keyword_count': total_count,
                    'relevance_score': relevance_score,
                    'keywords': keywords[:5],
                    'percentage': 0
                })

        # Sort by keyword count
        topics_data.sort(key=lambda x: x['keyword_count'], reverse=True)

        # Calculate percentages
        total_keywords = sum(topic['keyword_count'] for topic in topics_data)
        for topic in topics_data:
            topic['percentage'] = (topic['keyword_count'] / total_keywords * 100) if total_keywords > 0 else 0

        # Enhanced word frequency analysis for word cloud
        words = text_lower.split()

        meaningful_words = []
        for word in words:
            if (len(word) > 4 and word.isalpha() and 
                word not in ['which', 'would', 'could', 'should', 'their', 'there', 'these', 'those', 
                           'where', 'while', 'during', 'through', 'within', 'across', 'between']):
                meaningful_words.append(word)

        word_freq = Counter(meaningful_words)

        # Word cloud and frequency data
        wordcloud_data = dict(word_freq.most_common(50))
        top_words = dict(word_freq.most_common(15))

        return {
            'total_topics': len(topics_data),
            'topics_data': topics_data,
            'word_frequency': top_words,
            'wordcloud_data': wordcloud_data,
            'analysis_timestamp': datetime.now().isoformat(),
            'bank_name': self.current_bank,
            'text_length': len(text),
            'total_keyword_mentions': total_keywords
        }

    def _generate_and_save_plots(self, results: Dict[str, Any]):
        """Generate and save ONLY 2 plots"""
        try:
            # 1. Word frequency bar chart
            if results.get('word_frequency'):
                fig1 = self._create_word_frequency_chart(results['word_frequency'])
                self.data_manager.save_plot(self.current_bank, 'word_frequency_analysis', fig1, 'matplotlib')
                plt.close(fig1)

            # 2. Word cloud plot
            if results.get('wordcloud_data'):
                fig2 = self._create_word_cloud_chart(results['wordcloud_data'])
                self.data_manager.save_plot(self.current_bank, 'word_cloud_visualization', fig2, 'matplotlib')
                plt.close(fig2)

            logger.info(f"Generated and saved 2 topic analysis plots for {self.current_bank}")

        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            st.warning(f"Analysis completed but plot generation had issues: {str(e)}")

    def _create_word_frequency_chart(self, word_freq: Dict[str, int]):
        """Create word frequency bar chart"""
        fig, ax = plt.subplots(figsize=(12, 8))

        words = list(word_freq.keys())
        frequencies = list(word_freq.values())

        # Create horizontal bar chart for better readability
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        bars = ax.barh(words, frequencies, color=colors, alpha=0.8)

        ax.set_title('Top Financial Terms Frequency Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency Count', fontsize=12)
        ax.set_ylabel('Terms', fontsize=12)

        # Add value labels on bars
        for bar, freq in zip(bars, frequencies):
            width = bar.get_width()
            ax.text(width + max(frequencies) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{freq}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_word_cloud_chart(self, wordcloud_data: Dict[str, int]):
        """Create word cloud visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))

        words = list(wordcloud_data.keys())[:30]  # Top 30 words
        frequencies = list(wordcloud_data.values())[:30]

        # Normalize frequencies for sizing
        max_freq = max(frequencies)

        # Create pseudo-random positions for words
        np.random.seed(42)  # For consistent layout
        x_positions = np.random.uniform(0, 10, len(words))
        y_positions = np.random.uniform(0, 8, len(words))

        # Plot words with size based on frequency
        for i, (word, freq) in enumerate(zip(words, frequencies)):
            # Size based on frequency (15-60 point sizes)
            size = 15 + (freq / max_freq) * 45

            # Color based on frequency (using colormap)
            color = plt.cm.viridis(freq / max_freq)

            ax.text(x_positions[i], y_positions[i], word, 
                   fontsize=min(size, 60), color=color, 
                   ha='center', va='center', fontweight='bold',
                   alpha=0.8)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_title('Financial Terms Word Cloud', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')  # Hide axes for clean word cloud look

        # Add subtle background
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout()
        return fig

    def _display_current_plots(self):
        """Display current analysis plots section"""
        st.markdown("### 📊 Current Plots")
        st.success("✅ **New analysis plots generated!**")

        # Get most recent plots
        available_plots = self.data_manager.get_available_plots(self.current_bank)

        if available_plots:
            topic_plots = {}
            for plot_name, plot_files in available_plots.items():
                if any(keyword in plot_name.lower() for keyword in ['word_frequency', 'word_cloud']):
                    topic_plots[plot_name] = plot_files

            if topic_plots:
                cols = st.columns(2)
                for i, (plot_name, plot_files) in enumerate(topic_plots.items()):
                    with cols[i % 2]:
                        st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
                        if plot_files:
                            try:
                                latest_plot = plot_files[0]  # Most recent
                                st.image(latest_plot, caption=f"Current: {plot_name}", use_container_width=True)
                            except Exception as e:
                                st.write(f"Plot file: {Path(latest_plot).name}")
        else:
            st.info("Run analysis to generate current plots.")

    def _display_current_results(self):
        """Display current analysis results - streamlined"""
        results = st.session_state.topic_results

        st.markdown("### 📊 Topic Analysis Results")

        # Key metrics (streamlined)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Topics Found", results.get('total_topics', 0))

        with col2:
            st.metric("📝 Key Terms", len(results.get('word_frequency', {})))

        with col3:
            st.metric("🔢 Total Mentions", results.get('total_keyword_mentions', 0))

        # Topics overview
        topics_data = results.get('topics_data', [])
        if topics_data:
            st.markdown("#### 🏷️ Financial Topics Identified")

            # Create DataFrame for better display
            topics_df = pd.DataFrame([{
                'Topic': topic['name'],
                'Mentions': topic['keyword_count'],
                'Percentage': f"{topic['percentage']:.1f}%"
            } for topic in topics_data[:8]])  # Show top 8

            st.dataframe(topics_df, use_container_width=True)

            # Condensed topic details
            with st.expander("🔍 Topic Details", expanded=False):
                for topic in topics_data[:3]:  # Show top 3
                    st.markdown(f"**📊 {topic['name']}:** {', '.join(topic['keywords'])} ({topic['keyword_count']} mentions)")

def run_topic_modeling_agent():
    """Entry point for topic modeling agent"""
    agent = TopicModelingAgent()
    agent.run()