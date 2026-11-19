"""Claim verification and stance classification engine with dynamic model discovery and cascade fallback."""
from __future__ import annotations
import os
import re
import json
from typing import List, Tuple, Optional

from core.models import EvidencePassage

NEGATION_WORDS = {
    "not", "no", "never", "insufficient", "without", "cannot",
    "fail", "fails", "failed", "failure", "doesn't", "didn't",
    "rarely", "unlikely", "neither", "nor",
}

CONTRADICTION_CUES = {
    "contrast", "contrary", "however", "in contrast", "despite",
    "whereas", "unlike", "refutes", "disputes", "instead",
    "incorrect", "flawed", "misleading",
}

OPPOSITION_PAIRS = [
    ({"fail", "fails", "failed", "failure"}, {"competitive", "superior", "succeeds", "success", "effective", "improves"}),
    ({"increase", "increases", "increased"}, {"decrease", "decreases", "decreased", "reduce", "reduces"}),
    ({"improve", "improves", "improved"}, {"worsen", "worsens", "degrades", "degrade"}),
]

# Dynamic model caches
_CACHED_GEMINI_MODELS: Optional[List[str]] = None
_CACHED_GROQ_MODELS: Optional[List[str]] = None


def get_dynamic_gemini_models() -> List[str]:
    """Dynamically queries Google GenAI API to discover active generation models."""
    global _CACHED_GEMINI_MODELS
    if _CACHED_GEMINI_MODELS:
        return _CACHED_GEMINI_MODELS

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        discovered = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            # Prioritize fast text models, avoid audio/tts/experimental
            if "flash" in name and "tts" not in name and "audio" not in name:
                discovered.append(name)
        
        # Sort so newer versions (2.5 -> 2.0 -> 1.5) are attempted first
        discovered.sort(reverse=True)
        if discovered:
            _CACHED_GEMINI_MODELS = discovered
            return discovered
    except Exception as e:
        print(f"[CiteVerify] Dynamic Gemini model discovery fallback: {e}")

    _CACHED_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    return _CACHED_GEMINI_MODELS


def get_dynamic_groq_models() -> List[str]:
    """Dynamically queries Groq API to discover active text models."""
    global _CACHED_GROQ_MODELS
    if _CACHED_GROQ_MODELS:
        return _CACHED_GROQ_MODELS

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b"]

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        discovered = []
        for m in client.models.list().data:
            mid = m.id
            if "whisper" not in mid and "guard" not in mid:
                discovered.append(mid)

        # Prioritize fast versatile models
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        ordered = [p for p in preferred if p in discovered] + [d for d in discovered if d not in preferred]
        if ordered:
            _CACHED_GROQ_MODELS = ordered
            return ordered
    except Exception as e:
        print(f"[CiteVerify] Dynamic Groq model discovery fallback: {e}")

    _CACHED_GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.8-27b"]
    return _CACHED_GROQ_MODELS


def classify_with_rules(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """Deterministic symbolic stance classifier."""
    if not passages or not passages[0].text.strip():
        return (
            "Unsupported",
            "No authoritative evidence passage retrieved for this citation.",
            "Low",
        )

    best_passage = passages[0]
    score = getattr(best_passage, 'similarity_score', getattr(best_passage, 'score', 0.0))
    evidence_lower = best_passage.text.lower()
    claim_lower = claim_text.lower()

    claim_tokens = set(re.findall(r"\b\w{3,}\b", claim_lower))
    evidence_tokens = set(re.findall(r"\b\w{3,}\b", evidence_lower))
    overlap = claim_tokens & evidence_tokens

    # 1. Contradiction Detection
    has_cue = any(cue in evidence_lower for cue in CONTRADICTION_CUES)
    claim_negated = any(w in claim_tokens for w in NEGATION_WORDS)
    evidence_negated = any(w in evidence_tokens for w in NEGATION_WORDS)
    polarity_conflict = claim_negated != evidence_negated

    opposition_conflict = False
    for group_a, group_b in OPPOSITION_PAIRS:
        if (claim_tokens & group_a and evidence_tokens & group_b) or (
            claim_tokens & group_b and evidence_tokens & group_a
        ):
            opposition_conflict = True
            break

    if score >= 0.12 and (polarity_conflict or opposition_conflict or has_cue):
        return (
            "Contradicted",
            f"Evidence contains conflicting assertions (similarity: {score:.2f}, polarity mismatch detected).",
            "High" if score >= 0.25 else "Medium",
        )

    # 2. Outdatedness Check
    if cited_year and cited_year.isdigit():
        age = 2026 - int(cited_year)
        evolving_topics = {"transformer", "deep learning", "llm", "neural", "benchmark", "accuracy", "state-of-the-art"}
        if age > 8 and any(t in claim_lower for t in evolving_topics):
            return (
                "Potentially outdated",
                f"Cited source is from {cited_year} ({age} years old) for an evolving topic.",
                "Medium",
            )

    # 3. Numeric Inconsistency
    claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text))
    evidence_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", best_passage.text))
    if claim_nums and evidence_nums and not (claim_nums & evidence_nums):
        return (
            "Partially supported",
            f"Numeric discrepancy between claim figures {claim_nums} and evidence figures {evidence_nums}.",
            "Medium",
        )

    # 4. Support Detection
    if score >= 0.28 and len(overlap) >= 3:
        return (
            "Supported",
            f"Strong semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms).",
            "High",
        )
    elif score >= 0.14 and len(overlap) >= 2:
        return (
            "Partially supported",
            f"Moderate semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms).",
            "Medium",
        )
    else:
        return (
            "Unsupported",
            f"Insufficient semantic alignment (similarity: {score:.2f}, {len(overlap)} shared terms).",
            "Low",
        )


def classify_with_gemini_dynamic(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """Attempts Gemini inference across dynamically discovered models with automatic model fallback."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    models_to_try = get_dynamic_gemini_models()
    best_passage = passages[0].text[:1500]

    system_instruction = (
        "You are an academic claim and citation verification expert conforming to research integrity standards.\n"
        "Compare the Claim against the Evidence Passage from the cited research paper.\n"
        "Classify the stance into exactly one verdict:\n"
        "- 'Supported': The evidence explicitly confirms the claim.\n"
        "- 'Partially supported': The evidence relates to the topic, but lacks specific details or sample scale.\n"
        "- 'Contradicted': The evidence disagrees, refutes, or shows opposite findings.\n"
        "- 'Potentially outdated': The cited paper is historically valid but superseded for modern assertions.\n"
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

    for model_name in models_to_try:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=model_name,
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
            reasoning = data.get("reasoning", f"Evaluated via Google {model_name}.")
            return verdict, reasoning, confidence
        except Exception as e:
            print(f"[CiteVerify] Gemini model {model_name} failed: {e}. Trying fallback model...")
            continue

    return "FALLBACK", "All dynamic Gemini models unavailable", "Low"


def classify_with_groq_dynamic(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """Attempts Groq inference across dynamically discovered models with automatic model fallback."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    models_to_try = get_dynamic_groq_models()
    best_passage = passages[0].text[:1500]

    system_prompt = (
        "You are an academic claim verification auditor.\n"
        "Compare the Claim against the Evidence Passage.\n"
        "Classify into one verdict: 'Supported', 'Partially supported', 'Contradicted', 'Potentially outdated', 'Unsupported'.\n"
        "Respond strictly in JSON format with keys 'verdict', 'confidence', 'reasoning'."
    )

    user_prompt = (
        f"Claim: \"{claim_text}\"\n"
        f"Cited Year: {cited_year or 'Unknown'}\n"
        f"Evidence Passage: \"{best_passage}\""
    )

    for model_name in models_to_try:
        try:
            from groq import Groq

            client = Groq(api_key=groq_api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content
            data = json.loads(raw)
            verdict = data.get("verdict", "Partially supported")
            confidence = data.get("confidence", "Medium")
            reasoning = data.get("reasoning", f"Evaluated via Groq {model_name}.")
            return verdict, reasoning, confidence
        except Exception as e:
            print(f"[CiteVerify] Groq model {model_name} failed: {e}. Trying fallback model...")
            continue

    return "FALLBACK", "All dynamic Groq models unavailable", "Low"


def classify_claim(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
    use_llm: bool = True,
) -> Tuple[str, str, str]:
    """
    Cascading NLI Verification:
    1. Dynamic Google Gemini (models discovered at runtime, e.g. gemini-2.5-flash -> gemini-2.0-flash)
    2. Dynamic Groq LPU (models discovered at runtime, e.g. llama-3.3-70b -> llama-3.1-8b)
    3. Deterministic Symbolic Rule Engine
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if gemini_key and use_llm:
        verdict, reasoning, confidence = classify_with_gemini_dynamic(claim_text, passages, cited_year)
        if verdict != "FALLBACK":
            return verdict, reasoning, confidence

    if groq_key and use_llm:
        verdict, reasoning, confidence = classify_with_groq_dynamic(claim_text, passages, cited_year)
        if verdict != "FALLBACK":
            return verdict, reasoning, confidence

    return classify_with_rules(claim_text, passages, cited_year)
