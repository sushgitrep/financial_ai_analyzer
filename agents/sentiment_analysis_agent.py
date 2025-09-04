"""
Sentiment Analysis Agent 
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
import re

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_manager import DataManager

logger = logging.getLogger(__name__)

class SentimentAnalysisAgent:
    """Final polished sentiment analysis agent"""

    def __init__(self):
        self.data_manager = DataManager()
        self.current_bank = st.session_state.get('current_bank')

        # Sentiment word dictionaries
        self.positive_words = {
            'strong', 'growth', 'increased', 'improved', 'successful', 'profitable', 'positive', 
            'excellent', 'robust', 'solid', 'outstanding', 'effective', 'efficient', 'optimistic',
            'confident', 'momentum', 'expansion', 'gains', 'benefits', 'opportunities', 'stability'
        }

        self.negative_words = {
            'declined', 'decreased', 'weak', 'challenging', 'difficult', 'loss', 'negative', 
            'poor', 'concerns', 'risks', 'volatile', 'uncertainty', 'problems', 'issues',
            'deteriorated', 'pressures', 'constraints', 'obstacles', 'headwinds', 'adverse'
        }

        self.emotion_keywords = {
            'confidence': ['confident', 'assure', 'certain', 'believe', 'trust', 'conviction'],
            'concern': ['concern', 'worry', 'anxious', 'uncertain', 'doubt', 'cautious'],
            'optimism': ['optimistic', 'hopeful', 'positive', 'encouraging', 'promising'],
            'caution': ['cautious', 'careful', 'prudent', 'conservative', 'measured'],
            'urgency': ['urgent', 'immediate', 'critical', 'pressing', 'priority']
        }

    def run(self):
        """Run polished sentiment analysis with correct order"""
        st.subheader("💭 Sentiment & Emotion Analysis")

        if not self.current_bank:
            st.error("❌ No bank selected. Please go to Bank Selection tab first.")
            return

        # Check prerequisites
        if 'document_data' not in st.session_state:
            st.warning("⚠️ Please process a document first in the Preprocessing tab")
            return

        bank_info = self.data_manager.get_bank_info(self.current_bank)
        st.info(f"**Analyzing Sentiment for:** {bank_info['name'] if bank_info else self.current_bank}")

        # 1. Analysis section FIRST
        st.markdown("### 💭 Run Sentiment Analysis")

        available_plots = self.data_manager.get_available_plots(self.current_bank)
        if available_plots:
            st.warning("⚠️ Running new analysis will overwrite previous sentiment plots")

        if st.button("💭 Run Sentiment Analysis", type="primary", use_container_width=True, key="run_sentiment_analysis_polished"):
            self._run_sentiment_analysis()

        # 2. Current Plots section SECOND (after running analysis)
        if 'sentiment_results' in st.session_state:
            st.markdown("---")
            self._display_current_plots()

        # 3. FIXED: Previous Analysis section THIRD (only one instance, no duplicates)
        if available_plots:
            st.markdown("---")
            self._show_previous_analysis_section(available_plots)

        # 4. Analysis results LAST
        if 'sentiment_results' in st.session_state:
            st.markdown("---")
            self._display_current_results()

    def _show_previous_analysis_section(self, available_plots: Dict):
        """Show previous analysis plots section - SINGLE INSTANCE ONLY"""
        st.markdown("### 📂 Previous Sentiment Plots")

        if st.button("📊 View Previous Plots", type="secondary", use_container_width=True, key="view_sentiment_plots_polished"):
            self._show_previous_plots(available_plots)

        # Show plot preview
        with st.expander("🖼️ Previous Sentiment Analysis Plots", expanded=False):
            self._show_previous_plots(available_plots)

    def _show_previous_plots(self, available_plots: Dict):
        """Display previous sentiment analysis plots"""
        sentiment_plots = {}
        for plot_name, plot_files in available_plots.items():
            if any(keyword in plot_name.lower() for keyword in ['sentiment', 'emotion', 'feeling', 'mood']):
                sentiment_plots[plot_name] = plot_files

        if sentiment_plots:
            cols = st.columns(2)
            for i, (plot_name, plot_files) in enumerate(sentiment_plots.items()):
                with cols[i % 2]:
                    st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
                    if plot_files:
                        try:
                            latest_plot = plot_files[0]
                            st.image(latest_plot, caption=f"Previous: {plot_name}", use_container_width=True)
                        except Exception as e:
                            st.write(f"Plot file: {Path(latest_plot).name}")
        else:
            st.info("No previous sentiment analysis plots found.")

    def _run_sentiment_analysis(self):
        """Run actual sentiment analysis"""
        try:
            with st.spinner("💭 Running comprehensive sentiment analysis..."):
                doc_data = st.session_state.document_data
                text = doc_data.get('cleaned_text', doc_data.get('text', ''))

                # Real sentiment analysis
                results = self._perform_real_sentiment_analysis(text)

                # Save results to session
                st.session_state.sentiment_results = results

                # Save to persistent storage (overwriting previous)
                self.data_manager.save_analysis_results(self.current_bank, 'sentiment_results', results)

                # Generate and save ONLY 2 plots
                self._generate_and_save_plots(results)

                st.success("💭 Sentiment analysis completed and saved!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Sentiment analysis error: {e}")

    def _perform_real_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform real sentiment analysis"""

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        # Analyze each sentence
        sentence_sentiments = []
        overall_positive = 0
        overall_negative = 0
        overall_neutral = 0

        emotion_counts = {emotion: 0 for emotion in self.emotion_keywords.keys()}

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Count positive and negative words
            pos_count = sum(1 for word in self.positive_words if word in sentence_lower)
            neg_count = sum(1 for word in self.negative_words if word in sentence_lower)

            # Determine sentence sentiment
            if pos_count > neg_count:
                sentiment = 'positive'
                overall_positive += 1
            elif neg_count > pos_count:
                sentiment = 'negative'
                overall_negative += 1
            else:
                sentiment = 'neutral'
                overall_neutral += 1

            sentence_sentiments.append({
                'text': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'sentiment': sentiment,
                'positive_words': pos_count,
                'negative_words': neg_count,
                'confidence': abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
            })

            # Count emotions
            for emotion, keywords in self.emotion_keywords.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    emotion_counts[emotion] += 1

        # Calculate overall sentiment
        total_sentences = len(sentences)
        if total_sentences > 0:
            positive_percentage = (overall_positive / total_sentences) * 100
            negative_percentage = (overall_negative / total_sentences) * 100
            neutral_percentage = (overall_neutral / total_sentences) * 100
        else:
            positive_percentage = negative_percentage = neutral_percentage = 0

        # Determine overall sentiment
        if positive_percentage > negative_percentage:
            overall_sentiment = 'Positive'
            overall_confidence = positive_percentage / 100
        elif negative_percentage > positive_percentage:
            overall_sentiment = 'Negative'
            overall_confidence = negative_percentage / 100
        else:
            overall_sentiment = 'Neutral'
            overall_confidence = neutral_percentage / 100

        # Find sentiment-rich sentences for examples
        positive_examples = [s for s in sentence_sentiments if s['sentiment'] == 'positive' and s['confidence'] > 0.3][:3]
        negative_examples = [s for s in sentence_sentiments if s['sentiment'] == 'negative' and s['confidence'] > 0.3][:3]

        return {
            'overall_sentiment': overall_sentiment,
            'overall_confidence': overall_confidence,
            'sentiment_distribution': {
                'positive': positive_percentage,
                'negative': negative_percentage,
                'neutral': neutral_percentage
            },
            'sentiment_counts': {
                'positive': overall_positive,
                'negative': overall_negative,
                'neutral': overall_neutral,
                'total': total_sentences
            },
            'emotion_analysis': emotion_counts,
            'sentence_analysis': sentence_sentiments[:20],
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'analysis_timestamp': datetime.now().isoformat(),
            'bank_name': self.current_bank,
            'text_length': len(text)
        }

    def _generate_and_save_plots(self, results: Dict[str, Any]):
        """Generate and save ONLY 2 plots"""
        try:
            # 1. Sentiment distribution bar chart
            fig1 = self._create_sentiment_distribution_chart(results)
            self.data_manager.save_plot(self.current_bank, 'sentiment_distribution_bar', fig1, 'matplotlib')
            plt.close(fig1)

            # 2. Sentiment confidence pie chart
            fig2 = self._create_sentiment_pie_chart(results)
            self.data_manager.save_plot(self.current_bank, 'sentiment_confidence_pie', fig2, 'matplotlib')
            plt.close(fig2)

            logger.info(f"Generated and saved 2 sentiment analysis plots for {self.current_bank}")

        except Exception as e:
            logger.error(f"Error generating sentiment plots: {e}")
            st.warning(f"Analysis completed but plot generation had issues: {str(e)}")

    def _create_sentiment_distribution_chart(self, results: Dict[str, Any]):
        """Create sentiment distribution bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        sentiments = ['Positive', 'Negative', 'Neutral']
        distribution = results['sentiment_distribution']
        percentages = [distribution['positive'], distribution['negative'], distribution['neutral']]

        # Color scheme
        colors = ['#2E8B57', '#DC143C', '#808080']  # Green, Red, Gray

        bars = ax.bar(sentiments, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_title('Financial Document Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sentiment Categories', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_ylim(0, max(percentages) * 1.1 + 10)

        # Add percentage labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _create_sentiment_pie_chart(self, results: Dict[str, Any]):
        """Create sentiment distribution pie chart"""
        fig, ax = plt.subplots(figsize=(8, 8))

        distribution = results['sentiment_distribution']

        labels = []
        sizes = []
        colors = []

        if distribution['positive'] > 0:
            labels.append(f"Positive\n{distribution['positive']:.1f}%")
            sizes.append(distribution['positive'])
            colors.append('#2E8B57')

        if distribution['negative'] > 0:
            labels.append(f"Negative\n{distribution['negative']:.1f}%")
            sizes.append(distribution['negative'])
            colors.append('#DC143C')

        if distribution['neutral'] > 0:
            labels.append(f"Neutral\n{distribution['neutral']:.1f}%")
            sizes.append(distribution['neutral'])
            colors.append('#808080')

        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='',
                                             startangle=90, textprops={'fontsize': 10})

            # Enhance text
            for text in texts:
                text.set_fontweight('bold')

        ax.set_title(f'Overall Sentiment: {results["overall_sentiment"]}', 
                    fontsize=16, fontweight='bold', pad=20)

        return fig

    def _display_current_plots(self):
        """Display current analysis plots section"""
        st.markdown("### 📊 Current Plots")
        st.success("✅ **New sentiment plots generated!**")

        # Get most recent plots
        available_plots = self.data_manager.get_available_plots(self.current_bank)

        if available_plots:
            sentiment_plots = {}
            for plot_name, plot_files in available_plots.items():
                if any(keyword in plot_name.lower() for keyword in ['sentiment_distribution', 'sentiment_confidence']):
                    sentiment_plots[plot_name] = plot_files

            if sentiment_plots:
                cols = st.columns(2)
                for i, (plot_name, plot_files) in enumerate(sentiment_plots.items()):
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
        """Display current sentiment analysis results - streamlined"""
        results = st.session_state.sentiment_results

        st.markdown("### 💭 Sentiment Analysis Results")

        # Overall sentiment display
        overall_sentiment = results.get('overall_sentiment', 'Unknown')
        confidence = results.get('overall_confidence', 0)

        # Color-code the overall sentiment
        if overall_sentiment == 'Positive':
            sentiment_color = '🟢'
        elif overall_sentiment == 'Negative':
            sentiment_color = '🔴'
        else:
            sentiment_color = '🔵'

        st.markdown(f"""
        #### {sentiment_color} Overall Sentiment: **{overall_sentiment}**
        **Confidence Level:** {confidence:.1%}
        """)

        # Key metrics (streamlined)
        col1, col2, col3 = st.columns(3)

        sentiment_counts = results.get('sentiment_counts', {})

        with col1:
            st.metric("😊 Positive", sentiment_counts.get('positive', 0))

        with col2:
            st.metric("😞 Negative", sentiment_counts.get('negative', 0))

        with col3:
            st.metric("😐 Neutral", sentiment_counts.get('neutral', 0))

        # Sentiment distribution (streamlined)
        distribution = results.get('sentiment_distribution', {})
        if distribution:
            st.markdown("#### 📈 Sentiment Distribution")

            df_sentiment = pd.DataFrame([
                {'Sentiment': 'Positive', 'Percentage': f"{distribution.get('positive', 0):.1f}%"},
                {'Sentiment': 'Negative', 'Percentage': f"{distribution.get('negative', 0):.1f}%"},
                {'Sentiment': 'Neutral', 'Percentage': f"{distribution.get('neutral', 0):.1f}%"}
            ])

            st.dataframe(df_sentiment, use_container_width=True, hide_index=True)

        # Example sentences (condensed)
        with st.expander("📝 Example Sentences", expanded=False):
            positive_examples = results.get('positive_examples', [])
            negative_examples = results.get('negative_examples', [])

            if positive_examples:
                st.markdown("**😊 Positive Examples:**")
                for example in positive_examples[:2]:  # Show only 2
                    st.markdown(f"• {example['text']}")

            if negative_examples:
                st.markdown("**😞 Negative Examples:**")
                for example in negative_examples[:2]:  # Show only 2
                    st.markdown(f"• {example['text']}")

def run_sentiment_analysis_agent():
    """Entry point for sentiment analysis agent"""
    agent = SentimentAnalysisAgent()
    agent.run()