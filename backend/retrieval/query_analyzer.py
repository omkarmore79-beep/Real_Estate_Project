"""
Query Analyzer — detects user intent and expands queries for multi-query retrieval.
Uses Groq LLM for generating semantic search variations and a rule-based classifier
for intent categorization.
"""

from __future__ import annotations

import logging
import os
import json
import re
from typing import Any
from groq import Groq
from config import LLM_MODEL

logger = logging.getLogger(__name__)

# Intent category keywords/phrases
_INTENT_RULES = {
    "Pricing": re.compile(r"\b(?:price|cost|rate|pricing|lakh|crore|cost\s*sheet|payment|installment)\b", re.IGNORECASE),
    "Amenities": re.compile(r"\b(?:amenity|amenities|clubhouse|gym|pool|garden|play|park|recreation)\b", re.IGNORECASE),
    "Floor Plan": re.compile(r"\b(?:floor\s*plan|unit\s*plan|flat\s*layout|apartment\s*plan|carpet\s*area|bhk|dimensions)\b", re.IGNORECASE),
    "Master Plan": re.compile(r"\b(?:master\s*plan|layout|site\s*layout|township|tower\s*layout)\b", re.IGNORECASE),
    "Location": re.compile(r"\b(?:location|connectivity|landmark|highway|metro|road|address|where)\b", re.IGNORECASE),
    "Legal": re.compile(r"\b(?:legal|rera|maharera|approval|noc|developer\s*registration|title)\b", re.IGNORECASE),
    "Builder": re.compile(r"\b(?:builder|developer|group|company|hiranandani|constructed\s*by|built\s*by)\b", re.IGNORECASE),
    "Comparison": re.compile(r"\b(?:compare|comparison|versus|vs|better|difference|different)\b", re.IGNORECASE),
    "Construction Status": re.compile(r"\b(?:construction|status|progress|update|completed|possession|handover)\b", re.IGNORECASE),
}

_client = None

def _get_groq_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        _client = Groq(api_key=api_key)
    return _client

def classify_query_intent(query: str) -> list[str]:
    """Classify user query into one or more categories based on keyword rules."""
    intents = []
    for label, pattern in _INTENT_RULES.items():
        if pattern.search(query):
            intents.append(label)
    if not intents:
        intents.append("General Details")
    return intents

def expand_query(query: str) -> list[str]:
    """
    Generate 3 alternative versions of the user's search query using the LLM.
    Improves RAG retrieval recall across synonyms.
    """
    # Exclude trivial queries from expansion
    if len(query.strip().split()) <= 1:
        return [query]

    prompt = f"""You are a real estate search engine optimizer.
Generate exactly 3 alternative search queries (synonyms, variations, or specific formulations) for the following user question:
"{query}"

Focus on technical real estate synonyms (e.g. "floor plan" -> "layout layout", "price" -> "cost sheet", "possession" -> "handover date").
Return the output ONLY as a valid JSON array of strings. Do not include any explanation or markdown wrapping.

Example output format:
["alternative query 1", "alternative query 2", "alternative query 3"]
"""
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        
        # Strip code block markers if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        expanded = json.loads(content)
        if isinstance(expanded, list) and all(isinstance(x, str) for x in expanded):
            # Include original query as the primary query
            results = [query] + [x.strip() for x in expanded if x.strip()]
            return list(dict.fromkeys(results)) # Deduplicate
    except Exception as exc:
        logger.warning("Failed to expand query using LLM: %s. Using single query.", exc)
        
    return [query]
