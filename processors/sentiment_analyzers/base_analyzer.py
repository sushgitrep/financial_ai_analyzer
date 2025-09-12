import re


class BaseAnalyzer:

    def analyze(self, text_sections: list[dict]) -> list:
        results = []
        for section in text_sections:
            speaker = section.get("speaker")
            try:
                result = self.classifier(section["speech"])
                results.append(result[0] | {"speaker": section["speaker"]})
            except RuntimeError as e:
                chunks = self.split_into_chunks(section["speech"])
                chunk_results = []
                for chunk in chunks:
                    result = self.classifier(chunk)
                    chunk_results.append(result[0])

                aggregated = self.aggregate_analysis_results(chunk_results)
                results.append(aggregated | {"speaker": section["speaker"]})

        return results

    def aggregate_results_per_speaker(self, results: dict) -> dict:
        # Aggregate results per speaker
        aggregated_results = dict()
        for speaker, res in results.items():
            aggregated_results[speaker] = self.aggregate_analysis_results(res)

        return aggregated_results

    def split_into_chunks(self, text: str, max_words: int = 300) -> list[str]:
        """
        Split text into chunks of sentences, each no longer than max_words.
        Sentences are detected by punctuation (., ?, !).
        """
        # Split into sentences using regex
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        chunks = []
        current_chunk = []
        current_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            # If adding this sentence would exceed max_words, start a new chunk
            if current_count + word_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_count += word_count

        # Add last chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def aggregate_analysis_results(self, results: list[dict]) -> dict:
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r["label"].lower()] += 1
        total = sum(counts.values())
        sentiment_scores = {k: v / total for k, v in counts.items()}

        max_label = max(sentiment_scores, key=sentiment_scores.get)
        return {"label": max_label, "score": sentiment_scores[max_label]}
