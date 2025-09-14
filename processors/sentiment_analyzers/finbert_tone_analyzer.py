from processors.sentiment_analyzers.base_analyzer import BaseAnalyzer
from transformers import pipeline


class FinBERTToneAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            truncation=False,
            return_all_scores=True,
        )
