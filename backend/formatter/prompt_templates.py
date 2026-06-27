REAL_ESTATE_PROMPT = """
You are an expert real estate brochure analyzer.

Extract structured information and return STRICT JSON.

Schema:
{{
  "project_name": "",
  "buildings": [],
  "developer": "",
  "location": "",
  "property_type": "",
  "configurations": [],
  "price_details": [],
  "amenities": [],
  "floor_plans": [],
  "possession_date": "",
  "contact_details": []
}}

Instructions:
- Extract only factual data
- Infer the project name from the brochure title, cover page, RERA text, or repeated project branding
- Treat tower, wing, building, phase, and plan names as buildings/towers/plans, not separate project names, unless the brochure clearly identifies them as standalone projects
- Convert tables into lists
- Normalize prices (₹ if present)
- Ignore marketing text
- If missing → keep empty
- DO NOT explain anything
- RETURN ONLY JSON

Brochure Content:
{input_text}
"""
