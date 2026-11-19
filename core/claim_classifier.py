"""Claim verification and stance classification engine with dynamic model auto-discovery and fast fallback."""
from __future__ import annotations
import os
import re
import json
import time
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

_ACTIVE_GROQ_MODEL: Optional[str] = None
_ACTIVE_GEMINI_MODEL: Optional[str] = None
_GEMINI_DISABLED_UNTIL: float = 0.0
_GROQ_DISABLED_UNTIL: float = 0.0


def get_working_groq_model() -> Optional[str]:
    """Dynamically finds the best responsive model on Groq and caches it."""
    global _ACTIVE_GROQ_MODEL, _GROQ_DISABLED_UNTIL
    if time.time() < _GROQ_DISABLED_UNTIL:
        return None
    if _ACTIVE_GROQ_MODEL:
        return _ACTIVE_GROQ_MODEL

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        models = [m.id for m in client.models.list().data if "whisper" not in m.id and "guard" not in m.id]
        
        # Test candidate models in order of priority
        candidates = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b", "llama-3.1-8b-instant"]
        ordered = [c for c in candidates if c in models] + [m for m in models if m not in candidates]

        for cand in ordered[:3]:
            try:
                client.chat.completions.create(
                    model=cand,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=3,
                    timeout=3.0,
                )
                _ACTIVE_GROQ_MODEL = cand
                print(f"[CiteVerify] Active Groq model verified: {_ACTIVE_GROQ_MODEL}")
                return _ACTIVE_GROQ_MODEL
            except Exception:
                continue
    except Exception as e:
        print(f"[CiteVerify] Groq model discovery notice: {e}")

    _GROQ_DISABLED_UNTIL = time.time() + 180
    return None


def get_working_gemini_model() -> Optional[str]:
    """Dynamically checks if an active Gemini generation model is accessible with current quota."""
    global _ACTIVE_GEMINI_MODEL, _GEMINI_DISABLED_UNTIL
    if time.time() < _GEMINI_DISABLED_UNTIL:
        return None
    if _ACTIVE_GEMINI_MODEL:
        return _ACTIVE_GEMINI_MODEL

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "flash" in name and "tts" not in name and "audio" not in name:
                try:
                    client.models.generate_content(model=name, contents="ping")
                    _ACTIVE_GEMINI_MODEL = name
                    print(f"[CiteVerify] Active Gemini model verified: {_ACTIVE_GEMINI_MODEL}")
                    return _ACTIVE_GEMINI_MODEL
                except Exception:
                    continue
    except Exception as e:
        print(f"[CiteVerify] Gemini quota/access status: {e}")

    # Temporarily bypass Gemini if no models respond to prevent request stalls
    _GEMINI_DISABLED_UNTIL = time.time() + 300
    return None


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
    score = getattr(best_passage, "similarity_score", getattr(best_passage, "score", 0.0))
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


def classify_with_groq(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """Evaluates claim stance using dynamic Groq model."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    model = get_working_groq_model()
    if not model:
        return classify_with_rules(claim_text, passages, cited_year)

    best_passage = passages[0].text[:1200]
    system_prompt = (
        "You are an academic claim verification auditor.\n"
        "Compare the Claim against the Evidence Passage from the cited paper.\n"
        "Classify into one verdict: 'Supported', 'Partially supported', 'Contradicted', 'Potentially outdated', 'Unsupported'.\n"
        "Respond strictly in JSON format with keys:\n"
        "{\"verdict\": \"...\", \"confidence\": 0.9, \"reasoning\": \"1-2 sentence explanation.\"}"
    )

    user_prompt = (
        f"Claim: \"{claim_text}\"\n"
        f"Cited Year: {cited_year or 'Unknown'}\n"
        f"Evidence Passage: \"{best_passage}\""
    )

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=4.0,
        )
        data = json.loads(completion.choices[0].message.content)
        verdict = data.get("verdict", "Supported")
        conf_raw = data.get("confidence", "High")
        confidence = "High" if str(conf_raw).lower() in ["high", "0.9", "1.0", "1", "0.85"] else "Medium"
        reasoning = data.get("reasoning", f"Evaluated via Groq ({model}).")
        return verdict, reasoning, confidence
    except Exception as e:
        print(f"[CiteVerify] Groq inference notice ({model}): {e}. Using rule fallback.")
        return classify_with_rules(claim_text, passages, cited_year)


def classify_with_gemini(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
) -> Tuple[str, str, str]:
    """Evaluates claim stance using dynamic Gemini model."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or not passages or not passages[0].text.strip():
        return classify_with_rules(claim_text, passages, cited_year)

    model = get_working_gemini_model()
    if not model:
        return classify_with_groq(claim_text, passages, cited_year)

    best_passage = passages[0].text[:1200]
    system_instruction = (
        "You are an academic claim and citation verification expert.\n"
        "Compare the Claim against the Evidence Passage.\n"
        "Classify into exactly one verdict: 'Supported', 'Partially supported', 'Contradicted', 'Potentially outdated', 'Unsupported'.\n"
        "Respond strictly in JSON format with keys: verdict, confidence, reasoning."
    )
    user_content = (
        f"Claim: \"{claim_text}\"\n"
        f"Cited Year: {cited_year or 'Unknown'}\n"
        f"Evidence Passage: \"{best_passage}\""
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model=model,
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
        reasoning = data.get("reasoning", f"Evaluated via Gemini ({model}).")
        return verdict, reasoning, confidence
    except Exception as e:
        print(f"[CiteVerify] Gemini inference notice: {e}. Falling back to Groq.")
        return classify_with_groq(claim_text, passages, cited_year)


def classify_claim(
    claim_text: str,
    passages: List[EvidencePassage],
    cited_year: str | None = None,
    use_llm: bool = True,
) -> Tuple[str, str, str]:
    """
    Stance classification cascade:
    1. Dynamic Gemini (if verified and active)
    2. Dynamic Groq (sub-second inference via openai/gpt-oss-20b or discovered model)
    3. Deterministic Symbolic Rule Engine (sub-millisecond offline guarantee)
    """
    if not use_llm:
        return classify_with_rules(claim_text, passages, cited_year)

    # 1. Try Groq first for sub-second speed if Gemini is circuit-tripped
    model_groq = get_working_groq_model()
    if model_groq:
        return classify_with_groq(claim_text, passages, cited_year)

    # 2. Try Gemini
    model_gemini = get_working_gemini_model()
    if model_gemini:
        return classify_with_gemini(claim_text, passages, cited_year)

    # 3. Deterministic Rule Engine
    return classify_with_rules(claim_text, passages, cited_year)
