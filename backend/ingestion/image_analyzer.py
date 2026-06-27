import base64
from io import BytesIO
import json
import os
import re

from groq import Groq


VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
VISION_MAX_IMAGE_SIDE = int(os.getenv("VISION_MAX_IMAGE_SIDE", "1400"))
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "900000"))

IMAGE_TYPES = (
    "Amenities",
    "Floor Plan",
    "Master Plan",
    "Location Plan",
    "Site Layout",
    "Gallery",
    "Elevation",
    "Exterior",
    "Interior",
    "Project View",
    "Tower Layout",
    "Logo",
    "Advertisement",
    "QR Code",
    "Contact Information",
    "RERA Information",
    "Payment Plan",
    "Brochure Page",
)

IMAGE_TYPE_BY_PAGE_TYPE = {
    "Location Plan": "Location Plan",
    "Amenities": "Amenities",
    "Floor Plan": "Floor Plan",
    "Master Plan": "Master Plan",
    "Contact Information": "Contact Information",
    "RERA Information": "RERA Information",
    "Payment Plan": "Payment Plan",
    "Brochure Page": "Brochure Page",
}


PAGE_TYPE_DEFINITIONS = [
    {
        "slug": "location_plan",
        "label": "Location Plan",
        "phrases": (
            "location plan",
            "connectivity",
            "nearby landmarks",
            "map not to scale",
            "airport",
            "metro",
            "railway station",
            "hospital",
            "school",
            "highway",
            "distance",
        ),
        "keywords": (
            "roads",
            "airport",
            "railway",
            "station",
            "hospital",
            "school",
            "metro",
            "highway",
            "landmark",
            "distance",
            "map",
            "connectivity",
        ),
        "implicit_tags": (
            "location",
            "location plan",
            "connectivity",
            "map",
            "nearby landmarks",
            "airport",
            "metro",
            "hospital",
            "school",
            "highway",
            "approach road",
            "directions",
        ),
    },
    {
        "slug": "amenities",
        "label": "Amenities",
        "phrases": (
            "amenities",
            "clubhouse",
            "gymnasium",
            "swimming pool",
            "children",
            "jogging track",
            "indoor games",
        ),
        "keywords": (
            "clubhouse",
            "gym",
            "pool",
            "kids",
            "play",
            "jogging",
            "games",
            "garden",
            "sports",
        ),
        "implicit_tags": (
            "amenities",
            "facilities",
            "clubhouse",
            "gym",
            "pool",
            "swimming pool",
            "kids play area",
            "children play area",
            "jogging track",
            "indoor games",
            "recreation",
        ),
    },
    {
        "slug": "floor_plan",
        "label": "Floor Plan",
        "phrases": (
            "floor plan",
            "unit plan",
            "apartment layout",
            "carpet area",
            "bedroom",
            "kitchen",
            "balcony",
            "toilet",
        ),
        "keywords": (
            "layout",
            "bedroom",
            "kitchen",
            "balcony",
            "dimension",
            "carpet",
            "bhk",
            "toilet",
            "living",
        ),
        "implicit_tags": (
            "floor plan",
            "layout",
            "unit plan",
            "apartment layout",
            "flat plan",
            "2 bhk",
            "3 bhk",
            "4 bhk",
            "dimensions",
            "carpet area",
        ),
    },
    {
        "slug": "master_plan",
        "label": "Master Plan",
        "phrases": (
            "master plan",
            "master layout plan",
            "township",
            "tower layout",
            "site layout",
            "internal road",
            "landscape",
        ),
        "keywords": (
            "tower",
            "towers",
            "internal",
            "roads",
            "landscape",
            "township",
            "site",
            "layout",
            "entry",
            "podium",
        ),
        "implicit_tags": (
            "master plan",
            "site layout",
            "township",
            "tower layout",
            "project layout",
            "internal roads",
            "landscape plan",
            "development plan",
        ),
    },
    {
        "slug": "contact_information",
        "label": "Contact Information",
        "phrases": (
            "contact",
            "phone",
            "mobile",
            "email",
            "website",
            "office address",
            "qr code",
            "sales office",
        ),
        "keywords": (
            "phone",
            "mobile",
            "email",
            "website",
            "address",
            "office",
            "qr",
            "call",
        ),
        "implicit_tags": (
            "contact",
            "contact information",
            "phone",
            "mobile",
            "email",
            "website",
            "qr code",
            "office address",
            "sales office",
        ),
    },
    {
        "slug": "rera_information",
        "label": "RERA Information",
        "phrases": (
            "rera",
            "maharera",
            "registration number",
            "government registration",
        ),
        "keywords": (
            "rera",
            "maharera",
            "registration",
            "registered",
            "certificate",
            "government",
        ),
        "implicit_tags": (
            "rera",
            "maharera",
            "registration",
            "registration number",
            "government registration",
            "rera id",
        ),
    },
    {
        "slug": "payment_plan",
        "label": "Payment Plan",
        "phrases": ("payment plan", "payment chart", "price chart", "cost sheet"),
        "keywords": ("payment", "price", "cost", "installment", "schedule", "amount"),
        "implicit_tags": ("payment plan", "price chart", "cost sheet", "installments"),
    },
    {
        "slug": "brochure_page",
        "label": "Brochure Page",
        "phrases": (),
        "keywords": (),
        "implicit_tags": ("brochure", "page", "project information"),
    },
]


def _page_text_map(text):
    pages = {}
    matches = list(re.finditer(r"--- Page (\d+) \([^)]+\) ---", text or ""))

    for index, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages[page] = text[start:end].strip()

    return pages


def _unique(items):
    values = []
    seen = set()
    for item in items or []:
        if item is None:
            continue
        value = str(item).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _definition_by_slug(slug):
    for definition in PAGE_TYPE_DEFINITIONS:
        if definition["slug"] == slug:
            return definition
    return PAGE_TYPE_DEFINITIONS[-1]


def _slug_from_label(label):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label or "").lower()).strip("_")
    aliases = {
        "location": "location_plan",
        "location_map": "location_plan",
        "connectivity": "location_plan",
        "amenity": "amenities",
        "facilities": "amenities",
        "unit_layout": "floor_plan",
        "apartment_layout": "floor_plan",
        "site_layout": "master_plan",
        "development_plan": "master_plan",
        "contact": "contact_information",
        "rera": "rera_information",
        "maharera": "rera_information",
    }
    normalized = aliases.get(normalized, normalized)

    valid_slugs = {definition["slug"] for definition in PAGE_TYPE_DEFINITIONS}
    return normalized if normalized in valid_slugs else "brochure_page"


def _normalize_image_type(value, page_type=None):
    normalized = str(value or "").strip().lower()
    aliases = {
        "amenity": "Amenities",
        "amenities": "Amenities",
        "facilities": "Amenities",
        "floor": "Floor Plan",
        "floor_plan": "Floor Plan",
        "unit_plan": "Floor Plan",
        "apartment_layout": "Floor Plan",
        "master_plan": "Master Plan",
        "master_layout": "Master Plan",
        "location": "Location Plan",
        "location_map": "Location Plan",
        "location_plan": "Location Plan",
        "connectivity": "Location Plan",
        "site_layout": "Site Layout",
        "site_plan": "Site Layout",
        "gallery": "Gallery",
        "photo": "Gallery",
        "elevation": "Elevation",
        "exterior": "Exterior",
        "interior": "Interior",
        "project_view": "Project View",
        "tower_layout": "Tower Layout",
        "logo": "Logo",
        "advertisement": "Advertisement",
        "ad": "Advertisement",
        "qr": "QR Code",
        "qr_code": "QR Code",
        "contact": "Contact Information",
        "contact_information": "Contact Information",
        "rera": "RERA Information",
        "rera_information": "RERA Information",
        "payment": "Payment Plan",
        "payment_plan": "Payment Plan",
    }

    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if normalized in aliases:
        return aliases[normalized]

    for image_type in IMAGE_TYPES:
        if normalized == re.sub(r"[^a-z0-9]+", "_", image_type.lower()).strip("_"):
            return image_type

    return IMAGE_TYPE_BY_PAGE_TYPE.get(page_type, "Brochure Page")


def _extract_entities(text):
    text = text or ""
    entities = []
    entities.extend(re.findall(r"\bP\d{11,}\b", text, flags=re.IGNORECASE))
    entities.extend(re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE))
    entities.extend(re.findall(r"(?:\+?91[\s-]?)?[6-9]\d{9}\b", text))
    entities.extend(re.findall(r"\b(?:https?://)?(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?", text, flags=re.IGNORECASE))
    return _unique(entities)


def _fallback_page_metadata(image, page_text):
    haystack = (page_text or "").lower()
    best_score = 0
    best_definition = _definition_by_slug("brochure_page")

    for definition in PAGE_TYPE_DEFINITIONS:
        if definition["slug"] == "brochure_page":
            continue

        phrase_score = sum(4 for phrase in definition["phrases"] if phrase in haystack)
        keyword_score = sum(1 for keyword in definition["keywords"] if keyword in haystack)
        score = phrase_score + keyword_score

        if score > best_score:
            best_score = score
            best_definition = definition

    page_type = best_definition["label"]
    image_type = _normalize_image_type(None, page_type)
    tags = _unique([*best_definition["implicit_tags"], *best_definition["keywords"]])
    entities = _extract_entities(page_text)
    description = (
        f"Page {image.get('page')} appears to be a {page_type.lower()} based on "
        "its extracted text and brochure context."
        if best_score
        else f"Page {image.get('page')} is a brochure page. Vision analysis was unavailable."
    )

    return {
        "page": image.get("page"),
        "page_type": page_type,
        "image_type": image_type,
        "description": description,
        "searchable_tags": tags,
        "detected_entities": entities,
        "extracted_text": page_text or "",
        "image_references": [image.get("image_id")] if image.get("image_id") else [],
        "analysis_source": "text_fallback",
    }


def _clean_llm_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _image_data_url(path):
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(
                (VISION_MAX_IMAGE_SIDE, VISION_MAX_IMAGE_SIDE),
                Image.Resampling.LANCZOS,
            )

            for quality in (75, 65, 55, 45):
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                payload = buffer.getvalue()
                if len(payload) <= VISION_MAX_IMAGE_BYTES or quality == 45:
                    encoded = base64.b64encode(payload).decode("ascii")
                    return f"data:image/jpeg;base64,{encoded}"

    except Exception as exc:
        print(f"VISION IMAGE COMPRESSION ERROR ({path}):", exc)

    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _vision_prompt(page_number, page_text):
    page_types = ", ".join(definition["label"] for definition in PAGE_TYPE_DEFINITIONS)
    image_types = ", ".join(IMAGE_TYPES)
    return f"""
Analyze this real estate brochure page semantically. Infer the purpose of the page
from visual content, icons, maps, diagrams, tables, floor layouts, QR codes, and any
visible text. Do not require headings to exist.

Choose one page_type from: {page_types}.
Choose one image_type from: {image_types}.

Classification guidance:
- Roads, airport, railway station, hospitals, schools, landmarks, distance markers,
  map graphics, metro, or highways => Location Plan.
- Clubhouse, gym, swimming pool, children's play area, jogging track, indoor games
  => Amenities.
- Apartment layout, dimensions, bedroom, kitchen, balcony, unit plan => Floor Plan.
- Multiple towers, internal roads, landscape, township or project layout => Master Plan.
- Phone number, email, QR code, website, office address => Contact Information.
- RERA registration number, MahaRERA registration, government registration IDs
  => RERA Information. Extract IDs even when they are not explicitly labeled.

Return only compact JSON with these keys:
page_type, image_type, description, searchable_tags, detected_entities, extracted_text.

Page number: {page_number}
Existing OCR/text extraction:
{page_text[:2500] if page_text else "No extracted text available."}
""".strip()


def _analyze_with_vision(image, page_text):
    local_path = image.get("local_path")
    if not local_path or not os.path.exists(local_path) or not os.getenv("GROQ_API_KEY"):
        return None

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _vision_prompt(image.get("page"), page_text)},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(local_path)},
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=900,
    )
    content = response.choices[0].message.content
    parsed = json.loads(_clean_llm_json(content))

    page_slug = _slug_from_label(parsed.get("page_type"))
    definition = _definition_by_slug(page_slug)
    image_type = _normalize_image_type(parsed.get("image_type"), definition["label"])
    tags = _unique(
        [
            *definition["implicit_tags"],
            *definition["keywords"],
            *(parsed.get("searchable_tags") or []),
        ]
    )
    extracted_text = parsed.get("extracted_text") or page_text or ""
    entities = _unique([*(parsed.get("detected_entities") or []), *_extract_entities(extracted_text)])

    return {
        "page": image.get("page"),
        "page_type": definition["label"],
        "image_type": image_type,
        "description": parsed.get("description") or f"Page {image.get('page')} is a {definition['label'].lower()}.",
        "searchable_tags": tags,
        "detected_entities": entities,
        "extracted_text": extracted_text,
        "image_references": [image.get("image_id")] if image.get("image_id") else [],
        "analysis_source": "vision_llm",
    }


def analyze_page(image, page_texts=None):
    page_text = (page_texts or {}).get(image.get("page"), "")

    try:
        metadata = _analyze_with_vision(image, page_text)
    except Exception as exc:
        print(f"VISION PAGE ANALYSIS ERROR ({image.get('image_id')}):", exc)
        metadata = None

    if metadata is None:
        metadata = _fallback_page_metadata(image, page_text)

    slug = _slug_from_label(metadata.get("page_type"))
    definition = _definition_by_slug(slug)

    return {
        **image,
        **metadata,
        "page_number": image.get("page"),
        "image_type": metadata.get("image_type") or _normalize_image_type(None, metadata.get("page_type")),
        "tags": metadata.get("searchable_tags", []),
        "type": slug,
        "keywords": metadata.get("searchable_tags") or list(definition["implicit_tags"]),
        "entities": metadata.get("detected_entities", []),
        "text": metadata.get("extracted_text", ""),
    }


def analyze_images(images, extracted_text=""):
    page_texts = _page_text_map(extracted_text)
    return [analyze_page(image, page_texts) for image in images]


def page_metadata_from_images(images):
    pages = []
    for image in images:
        pages.append(
            {
                "page": image.get("page"),
                "page_number": image.get("page"),
                "page_type": image.get("page_type"),
                "image_type": image.get("image_type"),
                "description": image.get("description", ""),
                "searchable_tags": image.get("searchable_tags", image.get("keywords", [])),
                "detected_entities": image.get("detected_entities", image.get("entities", [])),
                "extracted_text": image.get("extracted_text", image.get("text", "")),
                "image_references": image.get("image_references") or [image.get("image_id")],
                "analysis_source": image.get("analysis_source", ""),
            }
        )
    return pages
