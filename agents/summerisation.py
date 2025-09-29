"""
GitHub Summarisation OpenAI ChatModel
=====================================

Project Plan - summarisation

We will develop map-reduce summarisation pipelines that generate structured outputs at
three levels: by topic (e.g., capital adequacy), by metric (e.g., CET1, NIM), and by speaker
(analyst vs management). Summaries will include citations linked back to transcript spans,
ensuring transparency and traceability. This hierarchical approach allows supervisors to
navigate insights from high-level themes down to specific evidence.
"""

# Setup
import os
from pathlib import Path

# OpenAI settings - Replace with your actual API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Folder paths - Set correct paths
source_path = ""  # path to folder containing earning call transcript pdfs
save_path = ""    # path to save summaries
test_file = ""    # test pdf file

# Install required packages (run this in terminal or uncomment and run once)
# pip install langchain_community langchain langchain-openai pypdf pdfplumber pdfminer.six tiktoken

# ────────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & IMPORTS
# ────────────────────────────────────────────────────────────────────────────────
import os
import json
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser

from langchain_community.document_loaders import PyPDFLoader
import pdfplumber

# ────────────────────────────────────────────────────────────────────────────────
# 2. PDF loading & chunking with overlap
# ────────────────────────────────────────────────────────────────────────────────
def load_pdf_texts(path: str):
    """
    Reuses a standard loader (PyPDFLoader) to extract page-wise docs, then merges to a single string.
    You can swap in your existing loader if you prefer (PDFMinerLoader, PDFPlumber, etc.).
    """
    docs = PyPDFLoader(path).load()
    # Join all page contents; keep page boundaries implicit
    full_text = "\n\n".join(d.page_content for d in docs)
    return full_text

def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
) -> List[str]:
    """
    Recursive splitter with overlap to avoid losing cross-boundary context.
    Tweak sizes to suit your model context & doc density.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], # coarse-to-fine
    )
    return splitter.split_text(text)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Pydantic schemas for structured, citable summaries
# ────────────────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A citation linking a summary point to a specific span of text."""
    text_span: str = Field(..., description="The exact text from the document that supports the summary point.")
    start_char: int = Field(..., description="The starting character index of the citation in the original chunk.")
    end_char: int = Field(..., description="The ending character index of the citation in the original chunk.")

class SpeakerSummary(BaseModel):
    """Key points and citations from a specific speaker (e.g., Management, Analyst)."""
    speaker: str = Field(..., description="The name or role of the speaker (e.g., 'Analyst 1', 'CEO').")
    key_points: List[str] = Field(..., description="Key points made by the speaker.")
    citations: List[Citation] = Field(..., description="Citations supporting the key points.")

class TopicSummary(BaseModel):
    """Summary of a specific financial topic with citations."""
    topic: str = Field(..., description="The financial topic (e.g., 'Revenue', 'Capital Adequacy').")
    summary: str = Field(..., description="A concise summary of the topic.")
    citations: List[Citation] = Field(..., description="Citations supporting the topic summary.")

class ChunkSummary(BaseModel):
    """Structured summary of a financial PDF chunk, with speaker and topic breakdowns."""
    topic_summaries: List[TopicSummary] = Field(..., description="Structured summaries of key financial topics.")
    speaker_summaries: List[SpeakerSummary] = Field(..., description="Structured summaries of key points by speaker.")

# The aggregate summary re-uses the same schema
AggregateSummary = ChunkSummary

# ────────────────────────────────────────────────────────────────────────────────
# 4. LLM + Prompt for structured chunk summaries
# ────────────────────────────────────────────────────────────────────────────────
chunk_parser = PydanticOutputParser(pydantic_object=ChunkSummary)

CHUNK_SUMMARY_TEMPLATE = """You are a precise financial analyst.
Summarize the following PDF chunk into the STRICT JSON schema provided.

The chunk is from a financial transcript and contains discussion between speakers (e.g., Management and Analysts).
Your task is to identify key financial topics and points made by each speaker.
For each summary point, you MUST provide a direct citation from the original chunk.
The citation should include the exact text and its character start and end indices within the chunk.

Topics to focus on:
- Capital Adequacy (e.g., CET1, leverage ratio)
- Earnings & Profitability (e.g., NIM, net income, non-interest income)
- Liquidity & Funding (e.g., deposits, LCR)
- Asset Quality & Credit Risk (e.g., NPLs, charge-offs)
- Key Risks & Outlook (e.g., regulatory, market, economic)

CHUNK:
----------------
{chunk}
----------------

{format_instructions}
"""

chunk_prompt = PromptTemplate(
    template=CHUNK_SUMMARY_TEMPLATE,
    input_variables=["chunk"],
    partial_variables={"format_instructions": chunk_parser.get_format_instructions()},
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def summarize_chunk(chunk_text: str) -> ChunkSummary:
    """Summarize a single chunk into the new structured Pydantic schema."""
    msg = chunk_prompt.format(chunk=chunk_text)
    resp = llm.invoke(msg)
    try:
        return chunk_parser.parse(resp.content)
    except Exception as e:
        print(f"Error parsing LLM response for chunk: {e}")
        return ChunkSummary(topic_summaries=[], speaker_summaries=[])

# ────────────────────────────────────────────────────────────────────────────────
# 5. Running merge of chunk-level summaries → aggregate summary
# ────────────────────────────────────────────────────────────────────────────────

def _merge_citations(citations_a: List[Citation], citations_b: List[Citation]) -> List[Citation]:
    """Merges and de-duplicates a list of citations."""
    seen = set()
    merged = []
    for cit in citations_a + citations_b:
        # Using a tuple of (text_span, start_char, end_char) for robust de-duplication
        citation_key = (cit.text_span, cit.start_char, cit.end_char)
        if citation_key not in seen:
            seen.add(citation_key)
            merged.append(cit)
    return merged

def _merge_topic_summaries(topics_a: List[TopicSummary], topics_b: List[TopicSummary]) -> List[TopicSummary]:
    """Merges and de-duplicates topics by topic name."""
    topic_map = {t.topic: t for t in topics_a}
    for t_new in topics_b:
        if t_new.topic in topic_map:
            # Merge summaries and citations for existing topic
            existing = topic_map[t_new.topic]
            # Concatenate summaries with a separator
            new_summary = f"{existing.summary} {t_new.summary}" if existing.summary and t_new.summary else existing.summary + t_new.summary
            merged_citations = _merge_citations(existing.citations, t_new.citations)
            topic_map[t_new.topic] = TopicSummary(topic=existing.topic, summary=new_summary, citations=merged_citations)
        else:
            # Add new topic
            topic_map[t_new.topic] = t_new
    return list(topic_map.values())

def _merge_speaker_summaries(speakers_a: List[SpeakerSummary], speakers_b: List[SpeakerSummary]) -> List[SpeakerSummary]:
    """
    Merges speaker summaries by speaker name.
    Assumes SpeakerSummary objects have 'speaker', 'key_points', and 'citations' fields
    as defined in the ChunkSummary schema.
    """
    speaker_map = {s.speaker: s for s in speakers_a}
    for s_new in speakers_b:
        if s_new.speaker in speaker_map:
            # Merge key points and citations for existing speaker
            existing = speaker_map[s_new.speaker]
            # Combine key points and de-duplicate
            merged_points = list(set(existing.key_points + s_new.key_points))
            merged_citations = _merge_citations(existing.citations, s_new.citations)
            # Create a new SpeakerSummary object with merged data
            speaker_map[s_new.speaker] = SpeakerSummary(
                speaker=existing.speaker,
                key_points=merged_points,
                citations=merged_citations
            )
        else:
            # Add new speaker
            speaker_map[s_new.speaker] = s_new
    return list(speaker_map.values())

def merge_summaries(running: AggregateSummary, new: ChunkSummary) -> AggregateSummary:
    """Combines structured summaries from two chunks."""
    merged_topics = _merge_topic_summaries(running.topic_summaries, new.topic_summaries)
    merged_speakers = _merge_speaker_summaries(running.speaker_summaries, new.speaker_summaries)
    return AggregateSummary(topic_summaries=merged_topics, speaker_summaries=merged_speakers)

# ────────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator: end-to-end PDF → aggregate structured summary (+ per-chunk)
# ────────────────────────────────────────────────────────────────────────────────
def summarize_pdf_to_structured_schema(
    pdf_path: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> Dict:
    """
    - Loads a PDF
    - Chunks with overlap
    - Summarizes each chunk into structured, citable summaries
    - Merges into a running/aggregate summary (same schema)
    Returns:
    {
        "pdf_path": ...,
        "chunk_size": ...,
        "chunk_overlap": ...,
        "chunk_summaries": [ {...}, ... ],
        "aggregate_summary": {...}
    }
    """
    text = load_pdf_texts(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunk_summaries: List[ChunkSummary] = []
    agg = AggregateSummary(topic_summaries=[], speaker_summaries=[])

    for i, ch in enumerate(chunks, start=1):
        cs = summarize_chunk(ch)
        chunk_summaries.append(cs)
        agg = merge_summaries(agg, cs)

    out = {
        "pdf_path": pdf_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_summaries": [cs.model_dump() for cs in chunk_summaries],
        "aggregate_summary": agg.model_dump(),
    }
    return out

# ────────────────────────────────────────────────────────────────────────────────
# 8.1 Second pass: condense to regulator-oriented key points
# ────────────────────────────────────────────────────────────────────────────────

# New Pydantic model for a key point with its source excerpt
class CitableKeyPoint(BaseModel):
    summary_point: str = Field(..., description="A concise summary point.")
    source_excerpt: str = Field(..., description="The excerpt from the original document that supports this point.")

class RegulatorKeyPoints(BaseModel):
    """
    Concise, regulator-facing takeaways with source excerpts.
    """
    executive_brief: str = Field(..., description="2–3 sentence summary prioritizing safety & soundness.")
    key_findings: List[CitableKeyPoint] = Field(..., description="Top 5–7 bullets with source excerpts. Capital, liquidity, asset quality, earnings, risk.")
    material_risks: List[CitableKeyPoint] = Field(..., description="3–5 bullets with source excerpts. Emerging/heightened risks incl. concentrations & conduct.")
    watch_metrics: List[CitableKeyPoint] = Field(..., description="Metrics to monitor with source excerpts.")
    suggested_followups: List[CitableKeyPoint] = Field(..., description="Concrete supervisor follow-ups / RFI asks with source excerpts.")

_reg_parser = PydanticOutputParser(pydantic_object=RegulatorKeyPoints)

REGULATOR_TEMPLATE = """You are a prudential supervisor synthesizing a report for regulators.
Your task is to generate a concise summary based on the provided structured aggregate summary from a financial transcript.
The input aggregate summary contains 'topic_summaries' and 'speaker_summaries', each with key points and citations.
Use the information in the 'topic_summaries' and 'speaker_summaries' fields to populate the output schema.
For each point in the output, include the relevant source excerpt from the original document, which is provided in the citations within the input.

Return ONLY the JSON described in the formatting instructions.

Guidance:
- Be precise, neutral, and evidence-based. No speculation beyond the text.
- Prefer short bullets (<= 15 words). Avoid repetition.
- Prioritize: capital adequacy, liquidity & funding, asset quality & concentrations, earnings durability, market/IRRBB,
  operational & conduct risk, macro/regulatory exposures, forward-looking guidance.
- If a detail is genuinely missing, omit it (de NOT invent).
- You MUST draw on the fields (topic_summaries, speaker_summaries) from the INPUT to extract key points and their source excerpts.
- Ensure the output strictly follows the RegulatorKeyPoints schema, including the 'summary_point' and 'source_excerpt' for each item in the lists.

INPUT (structured aggregate summary; JSON):
--------------------------
{aggregate_summary}
--------------------------

{format_instructions}
"""

_reg_prompt = PromptTemplate(
    template=REGULATOR_TEMPLATE,
    input_variables=["aggregate_summary"],
    partial_variables={"format_instructions": _reg_parser.get_format_instructions()},
)

def map_citations_to_text(full_text: str, summaries: List[dict]):
    """
    Given the full text of the document and a list of summaries with citations,
    this function adds the source excerpt to each citation.
    """
    for summary in summaries:
        if "citations" in summary:
            for citation in summary["citations"]:
                start = citation["start_char"]
                end = citation["end_char"]
                # Ensure start and end are within the bounds of full_text
                if 0 <= start < end <= len(full_text):
                    citation["source_excerpt"] = full_text[start:end]
                else:
                    citation["source_excerpt"] = "Invalid citation range"
                    print(f"Warning: Invalid citation range found: start={start}, end={end}, full_text_len={len(full_text)}")
    return summaries

def condense_for_regulator(
    aggregate_summary: dict,
    full_pdf_text: str,  # ADDED: new input parameter
    llm_instance: ChatOpenAI | None = None
) -> RegulatorKeyPoints:
    """
    Second-pass condensation into regulator key points.
    Args:
        aggregate_summary: dict from your first pass (agg.model_dump()).
        full_pdf_text: The full text of the original PDF.
        llm_instance: optionally pass your existing llm to reuse config.
    Returns:
        RegulatorKeyPoints pydantic object.
    """
    # Prepare the input for the LLM
    # Ensure the aggregate_summary dict is used directly as input
    agg_text = json.dumps(aggregate_summary, ensure_ascii=False, indent=2)

    _llm = llm_instance or llm
    msg = _reg_prompt.format(aggregate_summary=agg_text)
    resp = _llm.invoke(msg)
    try:
        return _reg_parser.parse(resp.content)
    except Exception as e:
        print(f"Error parsing RegulatorKeyPoints LLM response: {e}")
        print(f"LLM Response Content:\n{resp.content}")
        raise # Re-raise the exception after printing details

# 8.2 Second pass: Condense Speaker Summaries

class ManagementResponse(BaseModel):
    name: List[str] | str = Field(..., description="Name and/or title (e.g., 'CEO Jamie Dimon').")
    points: List[str] = Field(..., description="Key responses/statements from this management speaker.")

class SpeakerQASummary(BaseModel):
    """Q&A condensation (distinct from first-pass SpeakerSummary)."""
    analyst_points: str = Field(..., description="One paragraph summarizing analysts' key questions/concerns.")
    management_responses: List[ManagementResponse] = Field(..., description="Responses by named management speakers.")

speaker_parser = PydanticOutputParser(pydantic_object=SpeakerQASummary)

SPEAKER_SUMMARY_TEMPLATE = """You are a financial analyst specializing in earnings calls.
Based on the provided aggregate summary, create a concise summary organized by speaker role.

- Analyst Points: one paragraph (no bullets).
- Management Responses: list per named speaker (name + title if available).

INPUT (structured aggregate summary):
--------------------------
{aggregate_summary}
--------------------------

{format_instructions}
"""

speaker_prompt = PromptTemplate(
    template=SPEAKER_SUMMARY_TEMPLATE,
    input_variables=["aggregate_summary"],
    partial_variables={"format_instructions": speaker_parser.get_format_instructions()},
)

def generate_speaker_summary(aggregate_summary: Dict[str, Any]) -> SpeakerQASummary:
    agg_text = json.dumps(aggregate_summary, ensure_ascii=False, indent=2)
    msg = speaker_prompt.format(aggregate_summary=agg_text)
    resp = llm.invoke(msg)
    return speaker_parser.parse(resp.content)

# ────────────────────────────────────────────────────────────────────────────────
# 9. Example: integrate with your existing pipeline
# ────────────────────────────────────────────────────────────────────────────────

def summarize_pdf_for_regulators(
    pdf_path: str, chunk_size: int = 2000, chunk_overlap: int = 200
) -> Dict:
    filename = Path(pdf_path).stem
    print(filename)

    full_text = load_pdf_texts(pdf_path)
    print(f"Loaded {len(full_text)} characters from {pdf_path}")

    chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Split into {len(chunks)} chunks")
    if len(chunks) > 20:
        print(chunks[20])

    chunk_summaries: List[ChunkSummary] = []
    print(f"Processing {len(chunks)} chunks...")
    agg = AggregateSummary(topic_summaries=[], speaker_summaries=[])
    print("Running first-pass...")
    print(agg)

    for i, ch in enumerate(chunks, start=1):
        cs = summarize_chunk(ch)
        chunk_summaries.append(cs)
        print(f"Processed chunk {i}/{len(chunks)}")
        agg = merge_summaries(agg, cs)

    result = {
        "pdf_path": pdf_path,
        "bank": filename,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_summaries": [cs.model_dump() for cs in chunk_summaries],
        "aggregate_summary": agg.model_dump(),
    }

    regulator_view = condense_for_regulator(result["aggregate_summary"], full_text)
    speaker_summaries = generate_speaker_summary(result["aggregate_summary"])

    out = {
        **result,
        "regulator_key_points": regulator_view.model_dump(),
        "speaker_summary": speaker_summaries.model_dump()
    }
    return out

# ────────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set your paths and API key before running
    if test_file:
        result = summarize_pdf_for_regulators(test_file)
        print("Available result keys:", result.keys())
    else:
        print("Please set the test_file path and your OpenAI API key before running.")
