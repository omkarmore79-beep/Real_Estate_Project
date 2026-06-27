import re


IMAGE_INTENT_WORDS = {
    "show",
    "image",
    "photo",
    "picture",
    "plan",
    "layout",
    "map",
    "chart",
    "amenities",
    "amenity",
    "connectivity",
    "development",
    "elevation",
}

TYPE_ALIASES = {
    "location_plan": (
        "location plan",
        "location",
        "nearby",
        "nearby landmarks",
        "map",
        "airport",
        "metro",
        "hospital",
        "school",
        "connectivity",
    ),
    "connectivity": ("connectivity", "airport", "metro", "highway", "corridor"),
    "development_plan": ("development plan", "development"),
    "floor_plan": ("floor plan", "floor", "unit plan", "flat plan", "apartment layout", "layout", "bhk"),
    "master_plan": ("master plan", "master layout", "project plan", "site layout", "township", "tower layout"),
    "amenities": ("amenities", "amenity", "facilities", "clubhouse", "gym", "pool", "kids play"),
    "elevation": ("elevation", "building view", "tower view"),
    "payment_plan": ("payment plan", "payment chart", "price chart", "cost chart"),
    "map": ("map", "maps", "route", "directions"),
    "site_layout": ("site layout", "layout", "site plan"),
    "brochure_banner": ("banner", "cover", "brochure image"),
    "table": ("table", "chart", "schedule"),
    "contact_information": ("contact", "phone", "mobile", "email", "website", "qr", "office address"),
    "rera_information": ("rera", "maharera", "registration", "registration number", "rera id"),
}

IMAGE_TYPE_TO_SLUGS = {
    "Amenities": {"amenities"},
    "Floor Plan": {"floor_plan"},
    "Master Plan": {"master_plan"},
    "Location Plan": {"location_plan"},
    "Site Layout": {"site_layout", "master_plan"},
    "Gallery": {"gallery"},
    "Elevation": {"elevation"},
    "Exterior": {"exterior_view", "project_view"},
    "Interior": {"interior_view"},
    "Project View": {"project_view", "exterior_view"},
    "Tower Layout": {"tower_layout", "master_plan", "site_layout"},
}


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def has_image_intent(question):
    question_lower = question.lower()
    return any(word in question_lower for word in IMAGE_INTENT_WORDS)


def should_prioritize_image(question):
    question_lower = question.lower()
    return any(
        phrase in question_lower
        for phrase in (
            "show",
            "image",
            "photo",
            "picture",
            "display",
            "view",
            "map",
            "chart",
        )
    )


def _requested_types(question):
    question_lower = question.lower()
    matches = []

    for image_type, aliases in TYPE_ALIASES.items():
        if any(alias in question_lower for alias in aliases):
            matches.append(image_type)

    return matches


def _metadata_text(image):
    values = [
        image.get("type", "").replace("_", " "),
        image.get("image_type", ""),
        image.get("page_type", ""),
        image.get("description", ""),
        image.get("text", ""),
        image.get("extracted_text", ""),
        " ".join(image.get("tags", [])),
        " ".join(image.get("keywords", [])),
        " ".join(image.get("searchable_tags", [])),
        " ".join(str(entity) for entity in image.get("entities", [])),
        " ".join(str(entity) for entity in image.get("detected_entities", [])),
    ]
    return " ".join(value for value in values if value).lower()


def _score_image(question, question_tokens, requested_types, image):
    score = 0
    image_type = image.get("type", "")
    classified_image_type = image.get("image_type", "")
    page_type = image.get("page_type", "").lower().replace(" ", "_")

    matching_slugs = IMAGE_TYPE_TO_SLUGS.get(classified_image_type, set())
    if image_type in requested_types or page_type in requested_types or matching_slugs.intersection(requested_types):
        score += 10

    searchable = _metadata_text(image)
    searchable_tokens = _tokens(searchable)

    score += len(question_tokens.intersection(searchable_tokens))

    for requested_type in requested_types:
        if requested_type.replace("_", " ") in searchable:
            score += 4

    if question.lower() in searchable:
        score += 2

    return score


def _images_by_reference(project):
    return {
        image.get("image_id"): image
        for image in project.get("images", [])
        if image.get("image_id")
    }


def _page_as_image(page, project_images):
    references = page.get("image_references") or []
    image = project_images.get(references[0]) if references else None
    if image is None:
        image = next(
            (
                candidate
                for candidate in project_images.values()
                if candidate.get("page") == page.get("page")
            ),
            {},
        )

    return {
        **image,
        "page": page.get("page", image.get("page")),
        "page_number": page.get("page_number", page.get("page", image.get("page"))),
        "image_type": page.get("image_type", image.get("image_type")),
        "page_type": page.get("page_type", image.get("page_type")),
        "description": page.get("description", image.get("description", "")),
        "searchable_tags": page.get("searchable_tags", image.get("searchable_tags", [])),
        "detected_entities": page.get("detected_entities", image.get("detected_entities", [])),
        "extracted_text": page.get("extracted_text", image.get("extracted_text", "")),
        "image_references": references or image.get("image_references", []),
        "type": image.get("type") or page.get("page_type", "").lower().replace(" ", "_"),
        "tags": image.get("tags") or page.get("searchable_tags", []),
        "keywords": image.get("keywords") or page.get("searchable_tags", []),
        "entities": image.get("entities") or page.get("detected_entities", []),
        "text": image.get("text") or page.get("extracted_text", ""),
    }


def _candidate_images(project):
    candidates = []
    seen = set()

    for image in project.get("images", []):
        key = image.get("image_path") or image.get("image_id") or (image.get("page"), "image")
        seen.add(key)
        candidates.append(image)

    project_images = _images_by_reference(project)
    for page in project.get("pages", []):
        image = _page_as_image(page, project_images)
        key = image.get("image_path") or image.get("image_id") or (image.get("page"), "page")
        if key in seen:
            continue
        seen.add(key)
        candidates.append(image)

    return candidates


def _matches_allowed_image_type(image, allowed_image_types):
    if not allowed_image_types:
        return False

    allowed = {str(image_type).lower() for image_type in allowed_image_types}
    allowed_slugs = set()
    for image_type in allowed_image_types:
        normalized_type = str(image_type)
        allowed_slugs.update(IMAGE_TYPE_TO_SLUGS.get(normalized_type, set()))
        allowed_slugs.add(re.sub(r"[^a-z0-9]+", "_", normalized_type.lower()).strip("_"))

    image_values = {
        str(image.get("image_type", "")).lower(),
        str(image.get("page_type", "")).lower(),
    }
    image_slugs = {
        str(image.get("type", "")).lower(),
        re.sub(r"[^a-z0-9]+", "_", str(image.get("page_type", "")).lower()).strip("_"),
        re.sub(r"[^a-z0-9]+", "_", str(image.get("image_type", "")).lower()).strip("_"),
    }
    return bool(allowed.intersection(image_values) or allowed_slugs.intersection(image_slugs))


def find_matching_images(question, projects, limit=4, allowed_image_types=None):
    if allowed_image_types is not None and not allowed_image_types:
        return []

    requested_types = _requested_types(question)
    question_tokens = _tokens(question)
    scored_images = []

    for project in projects:
        for image in _candidate_images(project):
            if allowed_image_types is not None and not _matches_allowed_image_type(image, allowed_image_types):
                continue

            score = _score_image(question, question_tokens, requested_types, image)
            if allowed_image_types is not None:
                score += 8
            if score > 0:
                scored_images.append((score, image))

    scored_images.sort(key=lambda item: item[0], reverse=True)

    seen_paths = set()
    matches = []
    for _, image in scored_images:
        path = image.get("image_path")
        if not path or path in seen_paths:
            continue

        seen_paths.add(path)
        matches.append(image)

        if len(matches) >= limit:
            break

    return matches


def image_answer_text(question, images):
    if not images:
        return None

    image_type = images[0].get("image_type") or images[0].get("page_type") or images[0].get("type", "image").replace("_", " ")

    if any(word in question.lower() for word in ("what", "explain", "describe")):
        description = images[0].get("description")
        if description:
            return f"{description}. Related image attached."

    description = images[0].get("description")
    if description:
        return f"Here is the {image_type}. {description}"

    return f"Here is the {image_type}."
