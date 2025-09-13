import re


class BaseAnalyzer:

    def analyze(self, text_sections: list[dict]) -> list:
        results = []
        for section in text_sections:
            try:
                result = self.classifier(section["speech"])
                results.append(self.create_output_entry(result[0], section))
            except RuntimeError as e:
                chunks = self.split_into_chunks(section["speech"])
                chunk_results = []
                for chunk in chunks:
                    result = self.classifier(chunk)
                    chunk_results.append(result[0])

                aggregated = self.aggregate_analysis_results(chunk_results)
                results.append(self.create_output_entry(aggregated, section))

        return results

    def create_output_entry(
        self, result: list[dict], section: dict, margin: float = 0.1
    ) -> dict:
        # Extract scores
        scores = {item["label"].lower(): item["score"] for item in result}

        # Top label
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_scores[0]

        # Top label
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1]

        # Check margin threshold
        if top_label == "neutral" and (top_score - second_score) < margin:
            top_label_final = "neutral (uncertain)"
        else:
            top_label_final = top_label

        # Polarity / leaning
        polarity = scores["positive"] - scores["negative"]
        if top_label == "neutral":
            if abs(polarity) < 0.05:
                leaning = "truly neutral"
            elif polarity > 0:
                leaning = "neutral leaning positive"
            else:
                leaning = "neutral leaning negative"
        else:
            leaning = top_label  # if clearly pos/neg, leaning is same

        return {
            "top_label": top_label_final,
            "leaning": leaning,
            "polarity": polarity,
            "scores": scores,
            "speaker": section.get("speaker"),
            "speech": section.get("speech"),
        }

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
        """
        Given a list of FinBERT outputs (list of list of dicts),
        returns average score per label.
        """
        n = len(results)
        label_sums = {}

        for result in results:
            for item in result:
                label = item["label"].lower()
                score = item["score"]
                label_sums[label] = label_sums.get(label, 0.0) + score

        # Compute average

        avg_scores_list = [
            {"label": label, "score": label_sums[label] / n} for label in label_sums
        ]
        return avg_scores_list
