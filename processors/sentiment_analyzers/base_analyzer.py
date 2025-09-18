from pydoc import text
import re
from transformers import pipeline


class BaseAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            max_length=400,
            truncation=True,
        )

    def analyze(self, text_sections: list[dict]) -> list:
        results = []
        for section in text_sections:
            speaker = section.get("speaker")
            speech = section.get("speech", "")
            print(
                f"Speaker: {speaker}, Section length: {len(speech)}, Preview: {speech[:100]}"
            )
            # Always chunk if text is too long
            tokens = self.classifier.tokenizer.encode(
                section["speech"], add_special_tokens=True
            )
            if len(tokens) > 512:
                chunks = self.split_into_chunks(
                    section["speech"], self.classifier.tokenizer, max_tokens=510
                )
                chunk_results = []
                for chunk in chunks:
                    result = self.classifier(chunk, truncation=True, max_length=400)
                    chunk_results.append(result[0])
                    print(
                        f"Speaker: {speaker}, Chunk length: {len(chunk)}, Preview: {chunk[:100]}"
                    )
                aggregated = self.aggregate_analysis_results(chunk_results)
                results.append(
                    {
                        "speaker": speaker,
                        "aggregated": aggregated,
                        "chunks": chunk_results,
                    }
                )
            else:
                result = self.classifier(section["speech"])
                results.append(
                    {"speaker": speaker, "aggregated": result[0], "chunks": [result[0]]}
                )
        return results

    def aggregate_results_per_speaker(self, results: dict) -> dict:
        # Aggregate results per speaker
        aggregated_results = dict()
        for speaker, res in results.items():
            aggregated_results[speaker] = self.aggregate_analysis_results(res)

        return aggregated_results

    def split_into_chunks(
        self, text: str, tokenizer, max_tokens: int = 508
    ) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            token_count = len(tokenizer.encode(sentence, add_special_tokens=False))

            # If this sentence alone is too long, split it by words
            if token_count > max_tokens:
                # FIRST: Save the current chunk before processing the long sentence
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_tokens = 0

                # THEN: Process the oversized sentence by splitting it
                words = sentence.split()
                sub_chunk = []
                sub_tokens = 0
                for word in words:
                    word_token_count = len(
                        tokenizer.encode(word, add_special_tokens=False)
                    )
                    if sub_tokens + word_token_count > max_tokens and sub_chunk:
                        sub_chunk_text = " ".join(sub_chunk)
                        sub_chunk_tokens = tokenizer.encode(
                            sub_chunk_text, add_special_tokens=False
                        )
                        if len(sub_chunk_tokens) > max_tokens:
                            sub_chunk_tokens = sub_chunk_tokens[:max_tokens]
                            sub_chunk_text = tokenizer.decode(sub_chunk_tokens)
                        chunks.append(sub_chunk_text)
                        sub_chunk = []
                        sub_tokens = 0
                    sub_chunk.append(word)
                    sub_tokens += word_token_count
                if sub_chunk:
                    sub_chunk_text = " ".join(sub_chunk)
                    sub_chunk_tokens = tokenizer.encode(
                        sub_chunk_text, add_special_tokens=False
                    )
                    if len(sub_chunk_tokens) > max_tokens:
                        sub_chunk_tokens = sub_chunk_tokens[:max_tokens]
                        sub_chunk_text = tokenizer.decode(sub_chunk_tokens)
                    chunks.append(sub_chunk_text)
                continue

            # If adding this sentence would exceed max_tokens, start a new chunk
            if current_tokens + token_count > max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                if len(chunk_tokens) > max_tokens:
                    chunk_tokens = chunk_tokens[:max_tokens]
                    chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += token_count

        # Add last chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            if len(chunk_tokens) > max_tokens:
                chunk_tokens = chunk_tokens[:max_tokens]
                chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def aggregate_analysis_results(self, results: list[dict]) -> dict:
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r["label"].lower()] += 1
        total = sum(counts.values())
        sentiment_scores = {k: v / total for k, v in counts.items()}

        max_label = max(sentiment_scores, key=sentiment_scores.get)
        return {"label": max_label, "score": sentiment_scores[max_label]}
