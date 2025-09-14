import streamlit as st
from pathlib import Path
import sys
import logging
import pandas as pd
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).parent.parent))
from utils.data_manager import DataManager
from processors.sentiment_analyzer_factory import SentimentAnalyzerFactory

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    """Sentiment analysis agent using a comprehensive suite of Plotly visualizations."""

    def __init__(self):
        self.data_manager = DataManager()
        self.current_bank = st.session_state.get("current_bank")
        self.config = self.data_manager.config

    def run(self):
        """Run sentiment analysis and display interactive plots."""
        st.subheader("💭 Sentiment & Emotion Analysis")

        if not self.current_bank:
            st.error("❌ No bank selected. Please go to Bank Selection tab first.")
            return

        if "document_data" not in st.session_state:
            st.warning("⚠️ Please process a document first in the Preprocessing tab")
            return

        bank_info = self.data_manager.get_bank_info(self.current_bank)
        st.info(f"**Analyzing Sentiment for:** {bank_info['name'] if bank_info else self.current_bank}")

        st.markdown("### 💭 Run Sentiment Analysis")

        available_plots = self.data_manager.get_available_plots(self.current_bank)
        if available_plots:
            st.warning("⚠️ Running new analysis will overwrite previous sentiment plots")

        if st.button("💭 Run Sentiment Analysis", type="primary", use_container_width=True):
            self._run_sentiment_analysis()

        # This section now correctly displays plots from the current run
        if "sentiment_plots" in st.session_state:
            self._display_current_plots()

        if available_plots:
            st.markdown("---")
            self._show_previous_analysis_section(available_plots)

        if "sentiment_results" in st.session_state:
            st.markdown("---")
            self._display_current_results()

    def _run_sentiment_analysis(self):
        try:
            with st.spinner("💭 Running model-based sentiment analysis..."):
                doc_data = st.session_state.document_data
                text_sections = doc_data.get("text_sections", [])

                if not text_sections:
                    st.error("❌ No text sections found. Please re-process the document.")
                    return

                # 1. Perform Analysis
                results = self._perform_model_sentiment_analysis(text_sections)
                st.session_state.sentiment_results = results
                self.data_manager.save_analysis_results(self.current_bank, "sentiment_results", results)

                # 2. Generate and Save Plots
                sentiment_data = results.get("sentiments", {})
                output_path_prefix = f"sentiment_analysis_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
                
                # This will hold the figures for immediate display
                st.session_state.sentiment_plots = {}

                for model_name, speakers_data in sentiment_data.items():
                    st.session_state.sentiment_plots[model_name] = {}
                    plot_configs = {
                        "overall_volume": self._create_overall_sentiment_volume_plot,
                        "ratio_scores": self._create_ratio_score_plot,
                        "sentiment_trend": self._create_sentiment_trend_plot,
                        "sentiment_distribution": self._create_sentiment_distribution_plot,
                        "sentiment_counts": self._create_sentiment_counts_plot,
                        "sentiment_tokens": self._create_sentiment_token_plot,
                    }

                    for plot_key, plot_func in plot_configs.items():
                        fig = plot_func(speakers_data, model_name)
                        # Store figure in session state for immediate display
                        st.session_state.sentiment_plots[model_name][plot_key] = fig
                        
                        # Save figure to disk for future sessions
                        plot_name = f"{output_path_prefix}_{model_name.replace('/', '_')}_{plot_key}"
                        self.data_manager.save_plot(self.current_bank, plot_name, fig, "plotly")

                logger.info(f"Generated and saved all Plotly sentiment plots for {self.current_bank}")
                st.success("💭 Sentiment analysis completed!")
                # No st.rerun() needed here, Streamlit's flow will handle the update
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Sentiment analysis error: {e}", exc_info=True)

    def _perform_model_sentiment_analysis(self, text_sections: list) -> Dict[str, Any]:
        """Performs sentiment analysis using configured models."""
        models = self.config["sentiment_analysis"]["models"]
        sentiments = {}
        for model in models:
            analyzer = SentimentAnalyzerFactory.create_analyzer(model)
            sentiment_results = analyzer.analyze(text_sections)
            sentiments[model] = sentiment_results

        return {
            "sentiments": sentiments,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "bank_name": self.current_bank,
        }

    # --- PLOTTING FUNCTIONS (Unchanged) ---
    def _create_overall_sentiment_volume_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates a single stacked bar chart summarizing total token volume for key speakers."""
        total_pos, total_neg, total_neu = 0, 0, 0
        for speaker, data in speakers_data.items():
            if speaker == 'Operator' or data.get('total_tokens', 0) < 20: continue
            total_pos += sum(c['tokens'] for c in data['chunks'] if c['label'] == 'positive')
            total_neg += sum(c['tokens'] for c in data['chunks'] if c['label'] == 'negative')
            total_neu += sum(c['tokens'] for c in data['chunks'] if c['label'] == 'neutral')

        if (total_pos + total_neg + total_neu) == 0: return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=['Overall Sentiment'], x=[total_pos], name='Positive', orientation='h', marker_color='#2ca02c'))
        fig.add_trace(go.Bar(y=['Overall Sentiment'], x=[total_neu], name='Neutral', orientation='h', marker_color='#808080'))
        fig.add_trace(go.Bar(y=['Overall Sentiment'], x=[total_neg], name='Negative', orientation='h', marker_color='#d62728'))
        fig.update_layout(barmode='stack', title=f'Overall Sentiment Volume (Key Speakers Only)<br><sup>Model: {model_name}</sup>', xaxis_title='Total Number of Tokens', yaxis_title=None, template='plotly_white', legend_title_text='Sentiment')
        return fig

    def _create_ratio_score_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates an interactive horizontal bar chart for the ratio_score."""
        plot_data = [{'speaker': s, 'ratio_score': d.get('ratio_score', 0.0)} for s, d in speakers_data.items() if d.get('total_tokens', 0) >= 20]
        if not plot_data: return go.Figure()
        df = pd.DataFrame(plot_data)
        fig = px.bar(df, x='ratio_score', y='speaker', orientation='h', color='ratio_score', color_continuous_scale='RdYlGn', range_color=[-1, 1], title=f'Net Sentiment Score per Speaker<br><sup>Model: {model_name}</sup>', labels={'ratio_score': 'Token-Weighted Ratio Score', 'speaker': 'Speaker'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Ratio Score (-1.0 to +1.0)', yaxis_title=None, coloraxis_showscale=False, template='plotly_white')
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
        return fig

    def _create_sentiment_trend_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates a line chart showing sentiment trend over the course of the call."""
        all_chunks = [chunk for data in speakers_data.values() for chunk in data.get('chunks', [])]
        if not all_chunks: return go.Figure()
        
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        scores = [sentiment_map[chunk['label']] * chunk['score'] for chunk in all_chunks]
        df = pd.DataFrame({'score': scores})
        df['rolling_avg'] = df['score'].rolling(window=5, center=True, min_periods=1).mean()

        fig = px.line(df, y='rolling_avg', title=f'Sentiment Trend Over Call<br><sup>Model: {model_name}</sup>', labels={'index': 'Chunk Sequence', 'value': 'Sentiment Score'})
        fig.update_layout(template='plotly_white', yaxis_title='Sentiment Score (Smoothed)')
        return fig

    def _create_sentiment_distribution_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates a box plot showing the distribution of sentiment scores per speaker."""
        plot_data = [{'speaker': s, 'score': c['score'], 'label': c['label']} for s, d in speakers_data.items() if d.get('total_tokens', 0) >= 20 for c in d.get('chunks', [])]
        if not plot_data: return go.Figure()
        df = pd.DataFrame(plot_data)
        fig = px.box(df, x='speaker', y='score', color='speaker', title=f'Sentiment Score Distribution per Speaker<br><sup>Model: {model_name}</sup>', labels={'score': 'Sentiment Confidence Score', 'speaker': 'Speaker'})
        fig.update_layout(template='plotly_white', showlegend=False, xaxis_tickangle=-45)
        return fig

    def _create_sentiment_counts_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates an interactive stacked bar chart for sentiment chunk counts."""
        plot_data = [{'speaker': s, 'positive': d.get('positive', 0), 'negative': d.get('negative', 0), 'neutral': d.get('neutral', 0)} for s, d in speakers_data.items() if d.get('total_tokens', 0) >= 20]
        if not plot_data: return go.Figure()
        df = pd.DataFrame(plot_data)
        df['total_chunks'] = df['positive'] + df['negative'] + df['neutral']
        df = df.sort_values(by='total_chunks', ascending=False)
        fig = px.bar(df, x='speaker', y=['positive', 'neutral', 'negative'], title=f'Sentiment Chunk Distribution per Speaker<br><sup>Model: {model_name}</sup>', labels={'value': 'Number of Analyzed Chunks', 'speaker': 'Speaker'}, color_discrete_map={'positive': '#2ca02c', 'neutral': '#808080', 'negative': '#d62728'})
        fig.update_layout(barmode='stack', xaxis_tickangle=-45, template='plotly_white')
        return fig

    def _create_sentiment_token_plot(self, speakers_data: dict, model_name: str) -> go.Figure:
        """Generates an interactive stacked bar chart for sentiment token volume."""
        plot_data = []
        for speaker, data in speakers_data.items():
            if data.get('total_tokens', 0) < 20: continue
            plot_data.append({
                'speaker': speaker,
                'positive': sum(c['tokens'] for c in data['chunks'] if c['label'] == 'positive'),
                'negative': sum(c['tokens'] for c in data['chunks'] if c['label'] == 'negative'),
                'neutral': sum(c['tokens'] for c in data['chunks'] if c['label'] == 'neutral')
            })
        if not plot_data: return go.Figure()
        df = pd.DataFrame(plot_data)
        df['total_tokens'] = df['positive'] + df['negative'] + df['neutral']
        df = df.sort_values(by='total_tokens', ascending=False)
        fig = px.bar(df, x='speaker', y=['positive', 'neutral', 'negative'], title=f'Sentiment Token Volume per Speaker<br><sup>Model: {model_name}</sup>', labels={'value': 'Number of Tokens', 'speaker': 'Speaker'}, color_discrete_map={'positive': '#2ca02c', 'neutral': '#808080', 'negative': '#d62728'})
        fig.update_layout(barmode='stack', xaxis_tickangle=-45, template='plotly_white')
        return fig

    # --- DISPLAY FUNCTIONS ---
    def _display_current_plots(self):
        """Displays the newly generated plots from session_state."""
        st.markdown("### 📊 Current Interactive Plots")
        
        plot_order = ["overall_volume", "ratio_scores", "sentiment_trend", "sentiment_distribution", "sentiment_counts", "sentiment_tokens"]
        
        for model_name, plots in st.session_state.sentiment_plots.items():
            st.markdown(f"#### Results for Model: `{model_name}`")
            for plot_key in plot_order:
                if plot_key in plots:
                    fig = plots[plot_key]
                    st.plotly_chart(fig, use_container_width=True)

    def _show_previous_analysis_section(self, available_plots: Dict):
        """Shows the section for viewing previously generated plots."""
        st.markdown("### 📂 Previous Sentiment Plots")
        with st.expander("🖼️ View Previous Sentiment Analysis Plots", expanded=False):
            self._show_previous_plots(available_plots)

    def _show_previous_plots(self, available_plots: Dict):
        """Displays plots from storage, handling Plotly JSON."""
        plot_keys = ["overall", "ratio", "trend", "distribution", "counts", "tokens"]
        sentiment_plots = {name: files for name, files in available_plots.items() if any(key in name.lower() for key in plot_keys)}
        if not sentiment_plots:
            st.info("No previous sentiment analysis plots found.")
            return
        
        sorted_plot_names = sorted(sentiment_plots.keys(), key=lambda name: next((i for i, key in enumerate(plot_keys) if key in name.lower()), 99))

        for plot_name in sorted_plot_names:
            plot_files = sentiment_plots[plot_name]
            st.markdown(f"**{plot_name.replace('_', ' ').title()}**")
            if plot_files:
                try:
                    fig = self.data_manager.load_plot(plot_files[0])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load plot {plot_name}: {e}")

    def _display_current_results(self):
        """Displays the detailed sentiment results in a sortable table."""
        results = st.session_state.sentiment_results
        st.markdown("### 💭 Detailed Sentiment Results")
        
        sentiment_data = results.get("sentiments", {})
        for model_name, speakers_data in sentiment_data.items():
            with st.expander(f"**Model: {model_name}**", expanded=True):
                df_data = [{'Speaker': s, 'Ratio Score': d.get('ratio_score', 0.0), 'Positive Chunks': d.get('positive', 0), 'Negative Chunks': d.get('negative', 0), 'Neutral Chunks': d.get('neutral', 0), 'Total Tokens': d.get('total_tokens', 0)} for s, d in speakers_data.items()]
                if not df_data:
                    st.info("No data to display.")
                    continue
                df = pd.DataFrame(df_data).sort_values(by="Ratio Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                    "Ratio Score": st.column_config.ProgressColumn("Ratio Score", help="Token-weighted net sentiment. Ranges from -1 (very negative) to +1 (very positive).", format="%.3f", min_value=-1, max_value=1),
                })

def run_sentiment_analysis_agent():
    agent = SentimentAnalysisAgent()
    agent.run()
