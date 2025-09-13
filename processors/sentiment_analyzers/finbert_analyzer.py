from processors.sentiment_analyzers.base_analyzer import BaseAnalyzer
from transformers import pipeline


class FinBERTAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=False,
            return_all_scores=True,
        )
