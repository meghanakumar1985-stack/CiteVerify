"""Claim verification and stance classification engine with dual Groq NLI and symbolic inference."""
from __future__ import annotations
import os
import re
from typing import List, Tuple

from core.models import EvidencePassage

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "insufficient",
    "without",
    "cannot",
    "fail",
    "fails",
    "failed",
    "failure",
    "doesn't",
    "didn't",
    "rarely",
    "unlikely",
    "neither",
    "nor",
}

CONTRADICTION_CUES = {
    "contrast",
    "contrary",
    "however",
    "in contrast",
    "despite",
    "whereas",
    "unlike",
    "refutes",
    "disputes",
    "instead",
    "incorrect",
    "flawed",
    "misleading",
}

OPPOSITION_PAIRS = [
    ({"fail", "fails", "failed", "failure"}, {"competitive", "superior", "succeeds", "success", "effective", "improves"}),
    ({"increase", "increases", "increased"}, {"decrease", "decreases", "decreased", "reduce", "reduces"}),
    ({"improve", "improves", "improved"}, {"worsen", "worsens", "degrades", "degrade"}),
]


def _get_tokens(text: str) -> set[str]:
    """Extracts meaningful alphabetic tokens of length 3 or more."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _detect_polarity_conflict(claim_tokens: set[str], passage_tokens: set[str]) -> bool:
    """Checks for negation mismatch or semantic antonym opposition between claim and evidence."""
    claim_has_neg = bool(claim_tokens.intersection(NEGATION_WORDS))
    passage_has_neg = bool(passage_tokens.intersection(NEGATION_WORDS))
    passage_has_contrast = bool(passage_tokens.intersection(CONTRADICTION_CUES))

    if claim_has_neg != passage_has_neg:
        return True

    if passage_has_contrast:
        return True

    # Check opposition pairs (e.g. claim asserts fail, evidence asserts competitive/superior)
    for set_a, set_b in OPPOSITION_PAIRS:
        if (claim_tokens.intersection(set_a) and passage_tokens.intersection(set_b)) or (
            claim_tokens.intersection(set_b) and passage_tokens.intersection(set_a)
        ):
            return True

    return False


OUTDATED_CUES = {"state of the art", "sota", "recent", "modern", "current", "leading"}

def _extract_numbers_and_percents(text: str) -> set[str]:
    """Finds numeric and percentage tokens (e.g. '30%', '95')."""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text.lower()))


def classify_with_rules(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """
    Rule-based stance classifier complying with Hackathon Problem Statement 03:
    Verdicts: Supported, Partially supported, Contradicted, Potentially outdated, Unsupported.
    Returns: (verdict, details, confidence)
      confidence in {"High", "Medium", "Low"}
    """
    if not passages or not passages[0].text.strip():
        return "Unsupported", "No evidence passages retrieved for this claim.", "Low"

    best_passage = passages[0]
    score = best_passage.score

    if score < 0.08:
        return (
            "Unsupported",
            f"Retrieval similarity score ({score:.2f}) is below the minimum evidence threshold (0.08).",
            "Low",
        )

    # 1. Outdated Citation Check
    if cited_year:
        try:
            year_int = int(cited_year)
            current_year = 2026
            age = current_year - year_int
            claim_lower = claim_text.lower()
            has_modern_cue = any(cue in claim_lower for cue in OUTDATED_CUES)
            if age >= 15 or (age >= 8 and has_modern_cue):
                return (
                    "Potentially outdated",
                    f"Cited source is from {cited_year} ({age} years old). The claim references modern or evolving concepts that may no longer be state-of-the-art.",
                    "Medium",
                )
        except ValueError:
            pass

    claim_tokens = _get_tokens(claim_text)
    passage_tokens = _get_tokens(best_passage.text)
    overlap = claim_tokens.intersection(passage_tokens)

    # 2. Numeric & Caveat Discrepancy Check (matches Hackathon Section 05 example)
    claim_nums = _extract_numbers_and_percents(claim_text)
    passage_nums = _extract_numbers_and_percents(best_passage.text)
    has_pilot_caveat = any(w in passage_tokens for w in {"pilot", "limited", "preliminary", "small", "unsubstantiated", "substantiate"})
    missing_nums = [n for n in claim_nums if n not in passage_nums and "%" in n]
    if (missing_nums or (has_pilot_caveat and "%" in claim_text)) and score >= 0.12:
        return (
            "Partially supported",
            "The paper discusses general improvements, but the specific quantitative percentage or sample scale is disputed, limited, or not substantiated by the evidence.",
            "Medium",
        )

    # 3. Contradiction Detection
    has_conflict = _detect_polarity_conflict(claim_tokens, passage_tokens)
    if has_conflict and score >= 0.14 and len(overlap) >= 2:
        return (
            "Contradicted",
            f"Evidence addresses the topic (similarity: {score:.2f}, {len(overlap)} shared terms) but asserts contrary findings or polarity.",
            "High",
        )

    # 4. Support Detection
    if score >= 0.28 and len(overlap) >= 3:
        return (
            "Supported",
            f"Strong semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms) with consistent findings.",
            "High",
        )
    elif score >= 0.14 and len(overlap) >= 2:
        return (
            "Partially supported",
            f"Moderate semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms). Evidence relates to the claim topic with caveats.",
            "Medium",
        )
    else:
        return (
            "Unsupported",
            f"Insufficient semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms). Evidence lacks direct substantiation.",
            "Low",
        )


def classify_with_gemini(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """
    Verifies claim stance using Google Gemini 2.0 Flash for scientific NLI.
    Provides 1,000,000 Tokens/Min (TPM) on free tier without rate throttling.
    Falls back gracefully to deterministic rule-based inference if key is unset or network fails.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    try:
        from google import genai
        from google.genai import types
        import json

        client = genai.Client(api_key=gemini_key)
        best_passage = passages[0].text[:1500]

        system_instruction = (
            "You are an academic claim and citation verification expert conforming to research integrity standards.\n"
            "Compare the Claim against the Evidence Passage from the cited research paper.\n"
            "Classify the stance into exactly one verdict:\n"
            "- 'Supported': The evidence explicitly confirms the claim.\n"
            "- 'Partially supported': The evidence relates to the topic, but lacks specific details, sample scale, or exact percentages.\n"
            "- 'Contradicted': The evidence disagrees, refutes, or shows opposite findings.\n"
            "- 'Potentially outdated': The cited paper is too old for the contemporary assertion.\n"
            "- 'Unsupported': The evidence does not address the claim.\n\n"
            "Respond strictly in JSON format with keys:\n"
            "{\n"
            "  \"verdict\": \"Supported | Partially supported | Contradicted | Potentially outdated | Unsupported\",\n"
            "  \"confidence\": \"High | Medium | Low\",\n"
            "  \"reasoning\": \"A concise 1-2 sentence explanation of why the source matches or does not match.\"\n"
            "}"
        )

        user_content = (
            f"Claim: \"{claim_text}\"\n"
            f"Cited Year: {cited_year or 'Unknown'}\n"
            f"Evidence Passage: \"{best_passage}\""
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        data = json.loads(response.text)
        verdict = data.get("verdict", "Partially supported")
        confidence = data.get("confidence", "Medium")
        reasoning = data.get("reasoning", "Evaluated via Gemini 2.0 Flash NLI engine.")
        return verdict, reasoning, confidence
    except Exception as e:
        verdict, details, confidence = classify_with_rules(claim_text, passages, cited_year)
        return verdict, f"{details} (Inference fallback: {str(e)[:40]})", confidence


def classify_with_groq(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """
    Verifies claim stance using Groq (llama-3.1-8b-instant).
    Acts as automatic fallback when Google Gemini is offline or unconfigured.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    try:
        from groq import Groq
        import json

        client = Groq(api_key=groq_api_key)
        best_passage = passages[0].text[:1200]

        system_prompt = (
            "You are an academic claim and citation verification expert conforming to research integrity standards.\n"
            "Compare the Claim against the Evidence Passage from the cited research paper.\n"
            "Classify the stance into exactly one verdict:\n"
            "- 'Supported': The evidence explicitly confirms the claim.\n"
            "- 'Partially supported': The evidence relates to the topic, but lacks specific details, sample scale, or exact percentages.\n"
            "- 'Contradicted': The evidence disagrees, refutes, or shows opposite findings.\n"
            "- 'Potentially outdated': The cited paper is too old for the contemporary assertion.\n"
            "- 'Unsupported': The evidence does not address the claim.\n\n"
            "Respond strictly in JSON format with keys:\n"
            "{\n"
            "  \"verdict\": \"Supported | Partially supported | Contradicted | Potentially outdated | Unsupported\",\n"
            "  \"confidence\": \"High | Medium | Low\",\n"
            "  \"reasoning\": \"A concise 1-2 sentence explanation of why the source matches or does not match.\"\n"
            "}"
        )

        user_content = (
            f"Claim: \"{claim_text}\"\n"
            f"Cited Year: {cited_year or 'Unknown'}\n"
            f"Evidence Passage: \"{best_passage}\""
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=5.0,
        )

        content = chat_completion.choices[0].message.content
        data = json.loads(content)
        verdict = data.get("verdict", "Partially supported")
        confidence = data.get("confidence", "Medium")
        reasoning = data.get("reasoning", "Evaluated via Groq Llama-3.1-8B NLI engine.")
        return verdict, reasoning, confidence
    except Exception as e:
        verdict, details, confidence = classify_with_rules(claim_text, passages, cited_year)
        return verdict, f"{details} (Groq fallback: {str(e)[:40]})", confidence


def classify_claim(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
    use_llm: bool = True,
) -> Tuple[str, str, str]:
    """
    Stance classification cascade:
    1. Primary: Google Gemini 2.0 Flash
    2. Fallback: Groq Llama-3.1-8B-Instant
    3. Tertiary: Symbolic Rule Engine
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if gemini_key and use_llm:
        verdict, reasoning, confidence = classify_with_gemini(claim_text, passages, cited_year)
        if "fallback" not in reasoning.lower():
            return verdict, reasoning, confidence

    if groq_key and use_llm:
        verdict, reasoning, confidence = classify_with_groq(claim_text, passages, cited_year)
        if "fallback" not in reasoning.lower():
            return verdict, reasoning, confidence

    return classify_with_rules(claim_text, passages, cited_year)
