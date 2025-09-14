from transformers import pipeline
from processors.sentiment_analyzers.Finbertanalyzer_2 import FinBERTAnalyzer
from processors.sentiment_analyzers.Finbertanalyzer_1 import FinBERTProsusAnalyzer

class SentimentAnalyzerFactory:
    @staticmethod
    def create_analyzer(model_name):
        print("[create_analyzer] got:", repr(model_name))
        if model_name == "yiyanghkust/finbert-tone":
            classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
            return FinBERTAnalyzer(classifier)
        elif model_name == "ProsusAI/finbert":
            classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            return FinBERTProsusAnalyzer(classifier)
        # Add other models as needed
        else:
            raise ValueError(f"Unknown model: {model_name}")