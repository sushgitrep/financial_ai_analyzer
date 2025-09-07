from processors.sentiment_analyzers.base_analyzer import BaseAnalyzer
from transformers import pipeline


class FinancialRoBERTaAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="soleimanian/financial-roberta-large-sentiment",
            truncation=False,
        )
