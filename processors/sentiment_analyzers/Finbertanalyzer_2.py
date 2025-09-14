import re
from transformers import pipeline
from .base_analyzer import BaseAnalyzer

class FinBERTAnalyzer(BaseAnalyzer):
    def __init__(self, classifier):
        # Build the pipeline here (self-contained)
        self.classifier = classifier
        self.tokenizer = self.classifier.tokenizer

    def split_into_chunks(self, text: str, max_tokens: int = 510, stride_tokens: int = 64) -> list[str]:
        """
        Sentence-aware token chunking with overlap.
        max_tokens < 512 to leave margin for [CLS]/[SEP].
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        def tok_len(s: str) -> int:
            return len(self.tokenizer.encode(s, add_special_tokens=False))

        chunks, current, cur_len = [], [], 0

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            s_len = tok_len(s)

            # If a single sentence is too long, hard-split by tokens
            if s_len > max_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current, cur_len = [], 0
                chunks.extend(self._hard_token_split(s, max_tokens, stride_tokens))
                continue

            if cur_len + s_len <= max_tokens:
                current.append(s)
                cur_len += s_len
            else:
                if current:
                    chunks.append(" ".join(current))
                # start new with overlap (reuse last short sentence if possible)
                if current and tok_len(current[-1]) <= stride_tokens:
                    current = [current[-1], s]
                    cur_len = tok_len(current[-1]) + s_len
                else:
                    current = [s]
                    cur_len = s_len

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]

    def _hard_token_split(self, text: str, max_tokens: int, stride_tokens: int) -> list[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        out = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            piece = ids[start:end]
            out.append(self.tokenizer.decode(piece, skip_special_tokens=True))
            if end == len(ids):
                break
            start = max(end - stride_tokens, end) if end - stride_tokens <= start else end - stride_tokens
        return out

    def analyze(self, text_sections: list[dict]) -> dict:
        """Analyzes text sections and aggregates sentiment by speaker."""
        speaker_results = {}

        # Step 1: Gather all chunks for each speaker from all their sections
        for section in text_sections:
            speech = section.get("speech", "")
            if not speech:
                continue
            
            speaker = section.get("speaker", "unknown")
            print(f"Processing section for speaker: {speaker}")

            # Initialize the speaker's entry if it's their first section
            if speaker not in speaker_results:
                speaker_results[speaker] = {"chunks": []}

            # Chunk the speech from the current section
            tokens = len(self.tokenizer.encode(speech, add_special_tokens=False))
            chunks = [speech] if tokens <= 510 else self.split_into_chunks(speech)

            for chunk in chunks:
                result = self.classifier(chunk, truncation=True, max_length=512)[0]
                chunk_tokens = len(self.tokenizer.encode(chunk, add_special_tokens=False))
                
                # Append the result to the speaker's list of all chunks
                speaker_results[speaker]["chunks"].append({
                    "label": result["label"].lower(),
                    "score": float(result["score"]),
                    "tokens": chunk_tokens,
                    "text": chunk,
                })

        # Step 2: After processing all sections, calculate final TOKEN-WEIGHTED stats
        for speaker, data in speaker_results.items():
            all_chunks = data.get("chunks", [])
            if not all_chunks:
                continue

            pos_chunks = [c for c in all_chunks if c["label"] == "positive"]
            neg_chunks = [c for c in all_chunks if c["label"] == "negative"]

            total_tokens = sum(c["tokens"] for c in all_chunks)
            
            data["positive"] = len(pos_chunks)
            data["negative"] = len(neg_chunks)
            data["neutral"] = len(all_chunks) - data["positive"] - data["negative"]
            data["total_chunks"] = len(all_chunks)
            data["total_tokens"] = total_tokens
            
            # Calculate TOKEN-WEIGHTED average scores
            pos_score_weighted_sum = sum(c["score"] * c["tokens"] for c in pos_chunks)
            neg_score_weighted_sum = sum(c["score"] * c["tokens"] for c in neg_chunks)
            
            total_pos_tokens = sum(c["tokens"] for c in pos_chunks)
            total_neg_tokens = sum(c["tokens"] for c in neg_chunks)

            data["positive_score"] = pos_score_weighted_sum / total_pos_tokens if total_pos_tokens > 0 else 0.0
            data["negative_score"] = neg_score_weighted_sum / total_neg_tokens if total_neg_tokens > 0 else 0.0
            
            # The new ratio_score now reflects the overall "sentiment energy" of the speaker
            data["ratio_score"] = (pos_score_weighted_sum - neg_score_weighted_sum) / total_tokens if total_tokens > 0 else 0.0

        return speaker_results