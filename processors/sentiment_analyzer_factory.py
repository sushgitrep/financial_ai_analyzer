from processors.sentiment_analyzers.finbert_analyzer import FinBERTAnalyzer
from processors.sentiment_analyzers.finbert_tone_analyzer import FinBERTToneAnalyzer
from processors.sentiment_analyzers.financial_roberta_analyzer import (
    FinancialRoBERTaAnalyzer,
)


class SentimentAnalyzerFactory:
    @staticmethod
    def create_analyzer(model: str):
        if model == "finbert":
            return FinBERTAnalyzer()
        elif model == "finbert-tone":
            return FinBERTToneAnalyzer()
        elif model == "financial-roberta":
            return FinancialRoBERTaAnalyzer()
        else:
            raise ValueError(f"Unknown model: {model}")
