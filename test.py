from processors.sentiment_analyzer_factory import SentimentAnalyzerFactory

# Example text sections
text_sections = [
    {"speaker": "Alice", "speech": "The market is looking positive today!"},
    {"speaker": "Bob", "speech": "I am worried about the negative outlook."}
]

# Test both models
for model_name in ["yiyanghkust/finbert-tone", "ProsusAI/finbert"]:
    analyzer = SentimentAnalyzerFactory.create_analyzer(model_name)
    results = analyzer.analyze(text_sections)
    print(f"\nResults for {model_name}:")
    print(results)