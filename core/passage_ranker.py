"""Step 5: Chunk evidence text into passages and rank by TF-IDF cosine similarity."""
from __future__ import annotations
import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.models import EvidencePassage

# Abbreviations to protect from false sentence splits
PROTECTED_ABBRS = ("et al.", "i.e.", "e.g.", "fig.", "ref.", "vs.", "no.", "dr.", "prof.")


def split_sentences_safely(text: str) -> List[str]:
    """Splits text into sentences while protecting scientific abbreviations."""
    # Temporarily mask protected abbreviations
    masked = text
    masks = {}
    for idx, abbr in enumerate(PROTECTED_ABBRS):
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        placeholder = f"__ABBR_{idx}__"
        masks[placeholder] = abbr
        masked = pattern.sub(placeholder, masked)

    raw_sentences = re.split(r"(?<=[.!?])\s+", masked)
    sentences = []
    for s in raw_sentences:
        # Restore masked abbreviations
        for placeholder, abbr in masks.items():
            s = s.replace(placeholder, abbr)
        s = s.strip()
        if len(s) > 10:
            sentences.append(s)

    return sentences


def chunk_text(text: str, sentences_per_chunk: int = 2) -> List[str]:
    """Splits evidence text into multi-sentence passages (2 sentences per chunk by default)."""
    sentences = split_sentences_safely(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def rank_passages(
    claim_text: str,
    evidence_text: str,
    ref_id: str = "b0",
    top_k: int = 3,
) -> List[EvidencePassage]:
    """
    Chunks evidence text and ranks passages using TF-IDF cosine similarity to the claim.
    Returns the top_k ranked EvidencePassage objects (top 2-3).
    """
    if not evidence_text or not evidence_text.strip():
        return []

    passages = chunk_text(evidence_text, sentences_per_chunk=2)
    if not passages:
        return []

    # If only 1 passage, score it directly with TfidfVectorizer
    passages_to_fit = passages if len(passages) > 1 else [passages[0], claim_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z0-9_-]{2,}\b",
    )

    try:
        matrix = vectorizer.fit_transform(passages)
        query = vectorizer.transform([claim_text])
        scores = cosine_similarity(query, matrix).ravel()

        order = np.argsort(scores)[::-1]
        results: List[EvidencePassage] = []
        for rank, idx in enumerate(order[:top_k], start=1):
            score = float(scores[idx])
            results.append(
                EvidencePassage(
                    passage_id=f"{ref_id}_p{rank}",
                    ref_id=ref_id,
                    text=passages[idx],
                    score=round(score, 3),
                )
            )
        return results
    except ValueError:
        # Fallback if vocabulary is empty or only stopwords
        words_claim = set(re.findall(r"\b[a-z]{3,}\b", claim_text.lower()))
        results: List[EvidencePassage] = []
        for rank, p in enumerate(passages[:top_k], start=1):
            words_pass = set(re.findall(r"\b[a-z]{3,}\b", p.lower()))
            overlap = len(words_claim.intersection(words_pass)) / max(len(words_claim), 1)
            results.append(
                EvidencePassage(
                    passage_id=f"{ref_id}_p{rank}",
                    ref_id=ref_id,
                    text=p,
                    score=round(overlap, 3),
                )
            )
        return results
