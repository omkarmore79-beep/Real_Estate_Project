import re


PROJECT_SCHEMA = {
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
    "contact_details": [],
}


def _as_list(value):
    if value in (None, "", {}, []):
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_space(value):
    return re.sub(r"\s+", " ", value).strip(" .:-")


def _page_text(raw_text, page_number):
    pattern = rf"--- Page {page_number} \([^)]+\) ---"
    match = re.search(pattern, raw_text)
    if not match:
        return ""

    next_match = re.search(r"--- Page \d+ \([^)]+\) ---", raw_text[match.end() :])
    end = match.end() + next_match.start() if next_match else len(raw_text)
    return raw_text[match.end() : end].strip()


def _extract_project_name(raw_text):
    first_page = _page_text(raw_text, 1)
    match = re.search(
        r"(GOLDEN\s+WILLOWS\s+HIRANANDANI\s+FORTUNE\s+CITY)(?:\s*[—-]\s*(PANVEL))?",
        first_page,
        flags=re.IGNORECASE,
    )
    if match:
        name = _clean_space(match.group(1)).upper()
        if match.group(2):
            return f"{name} - {match.group(2).upper()}"
        return name

    match = re.search(r"([A-Z][A-Z\s]+HIRANANDANI\s+FORTUNE\s+CITY)", raw_text)
    if match:
        name = _clean_space(match.group(1)).upper()
        if re.search(r"\bPanvel\b", raw_text, flags=re.IGNORECASE):
            return f"{name} - PANVEL"
        return name

    return ""


def _extract_buildings(raw_text):
    buildings = []

    rera_section_match = re.search(
        r"The project has been registered via MahaRERA registration number:\s*(.*?)\s+and is available",
        raw_text,
        flags=re.IGNORECASE,
    )
    if rera_section_match:
        buildings.extend(re.findall(r"\b([A-Za-z]+):\s*P\d+", rera_section_match.group(1)))

    if not buildings:
        page_11 = _page_text(raw_text, 11)
        known_buildings = (
            "Iris",
            "Orchid",
            "Lavender",
            "Jasmine",
            "Aster",
            "Zenia",
            "Marigold",
            "Mayflower",
            "Acacia",
        )
        for building in known_buildings:
            if re.search(rf"\b{re.escape(building)}\b", page_11, flags=re.IGNORECASE):
                buildings.append(building)

    return list(dict.fromkeys(building.title() for building in buildings))


def _extract_developer(raw_text):
    if re.search(r"\bHiranandani Group\b", raw_text, flags=re.IGNORECASE):
        return "Hiranandani Group"
    return ""


def _extract_location(raw_text):
    if re.search(r"\bPanvel\b", raw_text, flags=re.IGNORECASE):
        return "Hiranandani Fortune City, Panvel, Maharashtra"
    return ""


def _extract_property_type(raw_text):
    if re.search(r"\bapartments?\b|\bflats?\b|\bhomes?\b", raw_text, flags=re.IGNORECASE):
        return "Apartments"
    return ""


def _extract_amenities(raw_text):
    amenities = []
    page_16 = _page_text(raw_text, 16)
    if page_16:
        page_16 = re.sub(r"^CLUBHOUSE AMENITIES\s*", "", page_16, flags=re.IGNORECASE)
        chunks = re.split(r"\s{2,}|(?<=[A-Z])\s+(?=[A-Z][A-Z’&/-]+(?:\s|$))", page_16)
        for chunk in chunks:
            item = _clean_space(chunk.title())
            if item and len(item) > 2:
                amenities.append(item)

    broader_patterns = [
        "Multi-cuisine Restaurants",
        "Retail Space",
        "Hiranandani Trust School",
        "Yotta Data Center Park",
        "Half-Olympic Sized Swimming Pool",
        "Children's Play Kingdom",
        "Badminton Courts",
        "Squash Courts",
        "Gymnasium",
        "Indoor Games",
        "Banquet/Party Hall",
        "Cafe Lounge",
        "Meditation Den",
        "Yoga Center",
        "Themed Spa & Massage",
        "Open Air Restaurant",
    ]
    raw_text_lower = raw_text.lower()
    for amenity in broader_patterns:
        if amenity.lower().replace("'", "’") in raw_text_lower or amenity.lower() in raw_text_lower:
            amenities.append(amenity)

    return list(dict.fromkeys(amenities))


def _extract_contacts(raw_text):
    contacts = []

    corporate_match = re.search(
        r"Corporate\s+.*?address\.?:\s*(.*?)\s+Tel\.?:\s*(.*?)\s+Site address:",
        raw_text,
        flags=re.IGNORECASE,
    )
    if corporate_match:
        phone_text = corporate_match.group(2)
        std_code_match = re.search(r"\(\+?(\d+)\s+(\d+)\)", phone_text)
        local_numbers = re.findall(r"\b\d{4}\s+\d{4}\b", phone_text)
        if std_code_match and local_numbers:
            country_code, city_code = std_code_match.groups()
            phones = [f"+{country_code} {city_code} {number}" for number in local_numbers]
        else:
            phones = re.findall(r"\+?\d[\d\s-]{6,}", phone_text)
        contacts.append(
            {
                "type": "Corporate Office",
                "address": _clean_space(corporate_match.group(1)),
                "phone": [_clean_space(phone) for phone in phones],
            }
        )

    site_match = re.search(
        r"Site address:\s*(.*?)\s+Call\s*:\s*(.*?)(?:\s+The project has been registered|\s+ORCHID MAHARERA|$)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if site_match:
        phones = re.findall(r"\+?\d[\d\s-]{6,}", site_match.group(2))
        contacts.append(
            {
                "type": "Site Office",
                "address": _clean_space(site_match.group(1)),
                "phone": [_clean_space(phone) for phone in phones],
            }
        )

    for tower, rera_id in re.findall(r"([A-Za-z]+):\s*(P\d+)", raw_text):
        contacts.append(
            {
                "type": "MahaRERA Registration",
                "tower_name": tower,
                "rera_id": rera_id,
            }
        )

    if not contacts:
        phones = re.findall(r"(?:\+?91[\s-]?)?[6-9]\d(?:[\s-]?\d){8}\b", raw_text)
        emails = re.findall(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            raw_text,
            flags=re.IGNORECASE,
        )
        websites = re.findall(
            r"\b(?:https?://)?(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:/[^\s|,]*)?",
            raw_text,
            flags=re.IGNORECASE,
        )
        address_patterns = [
            r"Corporate\s+Of(?:fi|ﬁ)ce\s+Address\s*[-:]\s*(.*?)(?=\s+Site\s+Address|\s+Disclaimer|$)",
            r"Site\s+Address\s*[-:]\s*(.*?)(?=\s+Disclaimer|$)",
            r"Postal\s+address\s*:\s*(.*?)(?=\s+(?:\+?91[\s-]?)?[6-9]\d|\s+www\.|\s+Disclaimer|$)",
        ]

        for phone in phones:
            contacts.append(_clean_space(phone))
        contacts.extend(_clean_space(email) for email in emails)
        contacts.extend(
            _clean_space(website).rstrip(".")
            for website in websites
            if "maharera" not in website.lower() and "gov.in" not in website.lower()
        )
        for pattern in address_patterns:
            for match in re.findall(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL):
                address = _clean_space(match)
                if address:
                    contacts.append(address)

    return list(dict.fromkeys(contacts))


def _extract_possession_date(raw_text):
    patterns = [
        r"\bpossession(?:\s+date)?\s*(?:is|:|-)?\s*([A-Za-z]+\s+\d{4})",
        r"\bpossession(?:\s+date)?\s*(?:is|:|-)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\b(?:completion|handover)\s*(?:date)?\s*(?:is|:|-)?\s*([A-Za-z]+\s+\d{4})",
        r"\b(?:completion|handover)\s*(?:date)?\s*(?:is|:|-)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\b(?:by|on or before)\s+([A-Za-z]+\s+\d{4})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return _clean_space(match.group(1))

    return ""


def normalize_project_data(data, raw_text):
    normalized = {}

    for key, default_value in PROJECT_SCHEMA.items():
        value = data.get(key, default_value) if isinstance(data, dict) else default_value
        if isinstance(default_value, list):
            value = _as_list(value)
        normalized[key] = value

    extracted_project_name = _extract_project_name(raw_text)
    if normalized["project_name"] in ("", "Unknown", None):
        normalized["project_name"] = extracted_project_name

    if not normalized["buildings"]:
        normalized["buildings"] = _extract_buildings(raw_text)

    if normalized["developer"] in ("", "Unknown", None):
        normalized["developer"] = _extract_developer(raw_text)

    if normalized["location"] in ("", "Unknown", None):
        normalized["location"] = _extract_location(raw_text)

    if normalized["property_type"] in ("", "Unknown", None):
        normalized["property_type"] = _extract_property_type(raw_text)

    if not normalized["amenities"]:
        normalized["amenities"] = _extract_amenities(raw_text)

    if not normalized["contact_details"]:
        normalized["contact_details"] = _extract_contacts(raw_text)

    if normalized["possession_date"] in ("", "Unknown", None):
        normalized["possession_date"] = _extract_possession_date(raw_text)

    if isinstance(data, dict) and data.get("images"):
        normalized["images"] = data["images"]

    return normalized
