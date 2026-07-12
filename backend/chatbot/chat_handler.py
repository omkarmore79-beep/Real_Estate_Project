import json
import os
import re

from groq import Groq

from storage.mongo_store import load_projects


client = None


def _get_groq_client():
    global client
    if client is None:
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    return client


def load_all_projects():
    return load_projects()


def _format_value(value):
    if value in (None, "", [], {}, "Unknown", "unknown"):
        return "Data not available"

    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return ", ".join(value)
        return json.dumps(value, indent=2)

    if isinstance(value, dict):
        return json.dumps(value, indent=2)

    return str(value)


def _is_missing_value(value):
    return value in (None, "", [], {}, "Unknown", "unknown")


def _clean_space(value):
    return re.sub(r"\s+", " ", str(value)).strip(" .:-")


def _extract_possession_date(raw_text):
    patterns = [
        r"\bpossession(?:\s+date)?\s*(?:is|:|-)?\s*([A-Za-z]+\s+\d{4})",
        r"\bpossession(?:\s+date)?\s*(?:is|:|-)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\b(?:completion|handover)\s*(?:date)?\s*(?:is|:|-)?\s*([A-Za-z]+\s+\d{4})",
        r"\b(?:completion|handover)\s*(?:date)?\s*(?:is|:|-)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\b(?:by|on or before)\s+([A-Za-z]+\s+\d{4})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text or "", flags=re.IGNORECASE)
        if match:
            return _clean_space(match.group(1))

    return ""


def _project_search_text(project):
    parts = [project.get("raw_text", "")]

    for page in project.get("pages", []):
        parts.extend(
            [
                page.get("page_type", ""),
                page.get("description", ""),
                page.get("extracted_text", ""),
                " ".join(str(entity) for entity in page.get("detected_entities", [])),
            ]
        )

    return "\n".join(part for part in parts if part)


def _extract_contact_details(text):
    text = text or ""
    contacts = []

    phones = re.findall(r"(?:\+?91[\s-]?)?[6-9]\d(?:[\s-]?\d){8}\b", text)
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE)
    websites = re.findall(
        r"\b(?:https?://)?(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:/[^\s|,]*)?",
        text,
        flags=re.IGNORECASE,
    )

    address_patterns = [
        r"Corporate\s+Of(?:fi|ﬁ)ce\s+Address\s*[-:]\s*(.*?)(?=\s+Site\s+Address|\s+Disclaimer|$)",
        r"Site\s+Address\s*[-:]\s*(.*?)(?=\s+Disclaimer|$)",
        r"Postal\s+address\s*:\s*(.*?)(?=\s+(?:\+?91[\s-]?)?[6-9]\d|\s+www\.|\s+Disclaimer|$)",
    ]

    addresses = []
    for pattern in address_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            address = _clean_space(match)
            if address:
                addresses.append(address)

    for phone in phones:
        contacts.append(_clean_space(phone))
    contacts.extend(_clean_space(email) for email in emails)
    contacts.extend(
        _clean_space(website).rstrip(".")
        for website in websites
        if "maharera" not in website.lower() and "gov.in" not in website.lower()
    )
    contacts.extend(addresses)

    return list(dict.fromkeys(item for item in contacts if item))


def _question_keywords(question):
    words = re.findall(r"[a-z0-9]+", question.lower())
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "available",
        "can",
        "does",
        "for",
        "from",
        "give",
        "have",
        "how",
        "is",
        "me",
        "of",
        "please",
        "project",
        "show",
        "the",
        "this",
        "to",
        "what",
        "where",
        "which",
    }
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    if any(word in question.lower() for word in ("location", "address", "where")):
        keywords.extend(["location", "address", "site", "near", "road", "city", "connectivity", "map", "landmark"])

    if "amenit" in question.lower() or "facilit" in question.lower():
        keywords.extend(["amenities", "facilities", "clubhouse", "gym", "pool"])

    if "floor plan" in question.lower() or "layout" in question.lower() or "bhk" in question.lower():
        keywords.extend(["floor plan", "layout", "unit plan", "apartment", "bedroom", "kitchen", "balcony", "bhk"])

    if "master plan" in question.lower() or "site plan" in question.lower():
        keywords.extend(["master plan", "site layout", "township", "tower", "internal road"])

    if "rera" in question.lower():
        keywords.extend(["rera", "maharera", "registration"])

    return list(dict.fromkeys(keywords))


def _extract_relevant_snippets(raw_text, question, limit=6):
    if not raw_text:
        return []

    keywords = _question_keywords(question)
    pages = []
    matches = list(re.finditer(r"--- Page (\d+) \([^)]+\) ---", raw_text))

    for index, match in enumerate(matches):
        page_number = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        text = re.sub(r"\s+", " ", raw_text[start:end]).strip()
        if not text:
            continue

        lowered = text.lower()
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            pages.append((score, int(page_number), text[:1200]))

    if not pages and raw_text:
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return [{"page": None, "text": cleaned[:1200]}] if cleaned else []

    pages.sort(key=lambda item: item[0], reverse=True)
    return [
        {"page": page_number, "text": text}
        for _, page_number, text in pages[:limit]
    ]


def _metadata_search_text(page):
    values = [
        page.get("page_type", ""),
        page.get("description", ""),
        page.get("extracted_text", ""),
        " ".join(page.get("searchable_tags", [])),
        " ".join(str(entity) for entity in page.get("detected_entities", [])),
    ]
    return " ".join(value for value in values if value).lower()


def _extract_relevant_pages(pages, question, limit=6):
    if not pages:
        return []

    keywords = _question_keywords(question)
    scored_pages = []

    for page in pages:
        searchable = _metadata_search_text(page)
        score = sum(1 for keyword in keywords if keyword.lower() in searchable)
        if question.lower() in searchable:
            score += 2
        if score:
            scored_pages.append((score, page.get("page") or 0, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "page": page.get("page"),
            "page_type": page.get("page_type"),
            "description": page.get("description"),
            "searchable_tags": page.get("searchable_tags", []),
            "detected_entities": page.get("detected_entities", []),
            "text": (page.get("extracted_text") or "")[:1200],
            "image_references": page.get("image_references", []),
        }
        for _, _, page in scored_pages[:limit]
    ]


def build_chat_context(projects, question):
    context = []

    for project in projects:
        item = dict(project)
        raw_text = item.pop("raw_text", "")
        item["relevant_text_snippets"] = _extract_relevant_snippets(raw_text, question)
        item["relevant_page_metadata"] = _extract_relevant_pages(
            item.get("pages", []),
            question,
        )
        context.append(item)

    return context


def _project_field(project, field):
    metadata = project.get("metadata") or {}

    if field == "project_name":
        return metadata.get("project") or project.get("project_name") or metadata.get("title")

    if field == "developer":
        return metadata.get("builder") or project.get("developer")

    value = project.get(field)

    if _is_missing_value(value) and field == "possession_date":
        return _extract_possession_date(_project_search_text(project))

    if _is_missing_value(value) and field == "contact_details":
        return _extract_contact_details(_project_search_text(project))

    return value


def _project_label(project, index):
    name = _project_field(project, "project_name")
    if name not in (None, "", "Unknown", "unknown"):
        return str(name)
    return f"Project {index + 1}"


def _matching_projects(question, projects):
    matches = []

    for project in projects:
        candidate_names = []

        name = _project_field(project, "project_name")
        if name and name not in ("Unknown", "unknown"):
            candidate_names.append(str(name))

        candidate_names.extend(str(building) for building in project.get("buildings", []))

        for item in project.get("contact_details", []):
            if isinstance(item, dict) and item.get("tower_name"):
                candidate_names.append(str(item["tower_name"]))

        normalized_names = {candidate.lower() for candidate in candidate_names if candidate}
        if any(candidate in question for candidate in normalized_names):
            matches.append(project)

    return matches or projects


def _matching_tower_names(question, project):
    names = []

    names.extend(str(building) for building in project.get("buildings", []))
    for item in project.get("contact_details", []):
        if isinstance(item, dict) and item.get("tower_name"):
            names.append(str(item["tower_name"]))

    normalized_matches = []
    for name in dict.fromkeys(names):
        if name.lower() in question:
            normalized_matches.append(name)

    return normalized_matches


def _format_project_field(projects, field):
    if field == "project_name":
        return "\n".join(
            _format_value(_project_field(project, "project_name")) for project in projects
        )

    values = []

    for index, project in enumerate(projects):
        label = _project_label(project, index)
        value = _format_value(_project_field(project, field))
        values.append(f"{label}: {value}")

    return "\n".join(values)


def answer_from_project_data(question, projects):
    question = question.lower()

    if not projects:
        return "No project data available. Please upload a brochure first."

    field_map = [
        (("location", "address", "where"), "location"),
        (("project name", "name of project"), "project_name"),
        (("building", "buildings", "tower", "towers", "plan names", "plans name"), "buildings"),
        (("developer", "builder"), "developer"),
        (("property type", "type of property"), "property_type"),
        (("configuration", "configurations", "bhk"), "configurations"),
        (("price", "cost", "rate"), "price_details"),
        (("amenity", "amenities", "facility", "facilities"), "amenities"),
        (("floor plan", "floor plans", "layout"), "floor_plans"),
        (("possession", "handover"), "possession_date"),
    ]

    for keywords, field in field_map:
        if any(keyword in question for keyword in keywords):
            matching_projects = _matching_projects(question, projects)

            if len(matching_projects) == 1:
                value = _project_field(matching_projects[0], field)
                if _is_missing_value(value):
                    return "Data not available"
                return _format_value(value)

            if all(_is_missing_value(_project_field(project, field)) for project in matching_projects):
                return "Data not available"

            return _format_project_field(matching_projects, field)

    if "rera" in question:
        matching_projects = _matching_projects(question, projects)

        if len(matching_projects) == 1:
            tower_names = _matching_tower_names(question, matching_projects[0])
            rera_items = [
                item
                for item in matching_projects[0].get("contact_details", [])
                if (
                    isinstance(item, dict)
                    and "rera_id" in item
                    and (
                        not tower_names
                        or str(item.get("tower_name", "")).lower()
                        in {name.lower() for name in tower_names}
                    )
                )
            ]
            return _format_value(rera_items)

        values = []
        for index, project in enumerate(matching_projects):
            tower_names = _matching_tower_names(question, project)
            rera_items = [
                item
                for item in project.get("contact_details", [])
                if (
                    isinstance(item, dict)
                    and "rera_id" in item
                    and (
                        not tower_names
                        or str(item.get("tower_name", "")).lower()
                        in {name.lower() for name in tower_names}
                    )
                )
            ]
            values.append(f"{_project_label(project, index)}: {_format_value(rera_items)}")

        return "\n".join(values)

    if any(
        word in question
        for word in ("contact", "phone", "mobile", "call", "office")
    ):
        matching_projects = _matching_projects(question, projects)

        if len(matching_projects) == 1:
            contact_details = _project_field(matching_projects[0], "contact_details")
            contacts = [
                item
                for item in contact_details
                if not (isinstance(item, dict) and "rera_id" in item)
            ]
            return _format_value(contacts)

        values = []
        for index, project in enumerate(matching_projects):
            contact_details = _project_field(project, "contact_details")
            contacts = [
                item
                for item in contact_details
                if not (isinstance(item, dict) and "rera_id" in item)
            ]
            values.append(f"{_project_label(project, index)}: {_format_value(contacts)}")

        return "\n".join(values)

    return None


def generate_answer(prompt):
    from config import LLM_MODEL
    try:
        response = _get_groq_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content

    except Exception as e:
        print("LLM ERROR:", e)

        if "429" in str(e) or "rate" in str(e).lower():
            return "Data not available"

        return "Data not available"
