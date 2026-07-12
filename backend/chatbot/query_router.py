"""
Query Router for the Industrial Maintenance Excavator RAG.
Classifies incoming technician queries into:
  - informational (basic descriptions/manual lookup)
  - diagnostic (symptom troubleshooting/DTC search)
  - procedural (maintenance instructions/torque specifications)
Extracts CAN bus DTC codes and component tags.
"""

from __future__ import annotations

import logging
import os
import json
import re
from typing import Any
from config import LLM_MODEL

logger = logging.getLogger(__name__)

# Fallback regex patterns for offline/safety parsing
DTC_PATTERN = re.compile(r"\b[ECA]-\d{3,4}\b|\b[ECA]\d{3,4}\b", re.IGNORECASE)

COMPONENT_KEYWORDS = {
    "hydraulic_pump": ["pump", "main pump", "hydraulic pump", "pumping"],
    "swing_motor": ["swing motor", "swing device", "swing mechanism"],
    "engine": ["engine", "excavator engine", "cummins", "motor"],
    "control_valve": ["mcv", "main control valve", "control valve", "spool"],
    "starter_motor": ["starter", "starter motor", "crank", "cranking"],
    "alternator": ["alternator", "battery charger", "generator"],
    "cylinders": ["cylinder", "boom cylinder", "arm cylinder", "bucket cylinder"],
}

_client = None

def _get_groq_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

def classify_im_query(query: str, groq_client=None) -> dict[str, Any]:
    """
    Classify a technician's maintenance query into informational, diagnostic, or procedural.
    Extracts referenced DTC codes and component tags.
    """
    # ── 1. Local Fallback parsing ───────────────────────────────────────────────
    dtc_codes = [code.upper() for code in DTC_PATTERN.findall(query)]
    components = []
    lower_query = query.lower()
    for comp, keywords in COMPONENT_KEYWORDS.items():
        if any(kw in lower_query for kw in keywords):
            components.append(comp)

    # ── 2. LLM Classification ───────────────────────────────────────────────────
    prompt = f"""You are an Industrial Maintenance AI Router for a Hyundai R215L excavator.
Analyze the technician's query and categorize it into:
- "informational": general descriptions, technical specs, parts details (e.g. "what is the hydraulic oil capacity")
- "procedural": step-by-step assembly, removal, adjustment, or repair guides (e.g. "how do I adjust swing play")
- "diagnostic": symptom-based troubleshooting, DTC/fault codes analysis (e.g. "engine cranks but won't start" or "DTC E-042 active")

Question: "{query}"

Output ONLY a valid JSON object matching this schema:
{{
  "category": "informational" | "diagnostic" | "procedural",
  "reason": "short explanation",
  "dtc_codes": ["DTC-1", "DTC-2"],
  "components": ["component_name_1"]
}}
"""
    try:
        client = groq_client or _get_groq_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean JSON markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)
        # Merge local regex extraction with LLM results
        result_dtc = list(set(result.get("dtc_codes", []) + dtc_codes))
        result_comp = list(set(result.get("components", []) + components))
        
        return {
            "category": result.get("category", "informational"),
            "reason": result.get("reason", ""),
            "dtc_codes": [x.upper().strip() for x in result_dtc if x.strip()],
            "components": [x.lower().strip() for x in result_comp if x.strip()],
        }
    except Exception as exc:
        logger.warning("LLM query routing failed: %s. Using keyword fallback classification.", exc)
        
    # Keyword/rules classification fallback
    category = "informational"
    if any(k in lower_query for k in ["fault", "symptom", "fail", "won't", "wont", "smoke", "noise", "pressure low", "code", "dtc"]):
        category = "diagnostic"
    elif any(k in lower_query for k in ["how do i", "how to", "procedure", "steps", "replace", "install", "remove", "adjust", "torque"]):
        category = "procedural"
        
    return {
        "category": category,
        "reason": "Rule-based fallback classification.",
        "dtc_codes": dtc_codes,
        "components": components,
    }
