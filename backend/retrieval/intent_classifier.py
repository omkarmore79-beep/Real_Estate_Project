import re


TEXT_ONLY_INTENTS = {
    "project_name": ("project name", "name of project", "project called"),
    "developer_name": ("developer", "builder", "developed by"),
    "rera_number": ("rera", "maharera", "registration number", "rera number", "rera id"),
    "contact_number": ("contact number", "phone", "mobile", "call", "telephone"),
    "email": ("email", "mail id", "e-mail"),
    "website": ("website", "web site", "url"),
    "office_address": ("office address", "site office", "sales office", "address"),
    "price": ("price", "cost", "rate", "pricing"),
    "payment_plan": ("payment plan", "payment schedule", "installment", "payment terms"),
    "unit_configuration": ("configuration", "configurations", "bhk", "unit configuration"),
    "specifications": ("specification", "specifications", "features"),
    "possession_date": ("possession", "handover"),
    "project_status": ("project status", "status"),
    "construction_update": ("construction", "construction update", "progress"),
    "legal_details": ("legal", "approval", "approvals", "title"),
}


VISUAL_INTENTS = {
    "amenities": {
        "phrases": ("amenities", "amenity", "facilities", "clubhouse", "gym", "pool", "kids play"),
        "image_types": {"Amenities"},
    },
    "floor_plan": {
        "phrases": ("floor plan", "unit plan", "flat plan", "apartment layout", "layout plan"),
        "image_types": {"Floor Plan"},
    },
    "master_plan": {
        "phrases": ("master plan", "master layout", "township layout", "project layout"),
        "image_types": {"Master Plan"},
    },
    "location_plan": {
        "phrases": ("location plan", "location map", "connectivity map", "nearby landmarks", "route map"),
        "image_types": {"Location Plan"},
    },
    "site_layout": {
        "phrases": ("site layout", "site plan"),
        "image_types": {"Site Layout", "Master Plan"},
    },
    "gallery": {
        "phrases": ("gallery", "photos", "pictures", "images"),
        "image_types": {"Gallery", "Project View", "Exterior", "Interior", "Elevation"},
    },
    "exterior_view": {
        "phrases": ("exterior", "external view", "outside view", "project view", "building view"),
        "image_types": {"Exterior", "Project View", "Elevation"},
    },
    "interior_view": {
        "phrases": ("interior", "inside view", "room view", "lobby"),
        "image_types": {"Interior"},
    },
    "elevation": {
        "phrases": ("elevation", "tower elevation", "building elevation"),
        "image_types": {"Elevation"},
    },
    "tower_layout": {
        "phrases": ("tower layout", "tower plan"),
        "image_types": {"Tower Layout", "Master Plan", "Site Layout"},
    },
}


VISUAL_ACTION_WORDS = {
    "show",
    "display",
    "view",
    "see",
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "pictures",
    "plan",
    "map",
    "layout",
}


def _contains_phrase(question, phrases):
    return any(phrase in question for phrase in phrases)


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def classify_intent(question):
    question_lower = (question or "").lower()
    question_tokens = _tokens(question_lower)

    for intent, phrases in TEXT_ONLY_INTENTS.items():
        if _contains_phrase(question_lower, phrases):
            return {
                "intent": intent,
                "requires_visual_response": False,
                "image_types": [],
            }

    for intent, config in VISUAL_INTENTS.items():
        if _contains_phrase(question_lower, config["phrases"]):
            return {
                "intent": intent,
                "requires_visual_response": True,
                "image_types": sorted(config["image_types"]),
            }

    if question_tokens.intersection(VISUAL_ACTION_WORDS):
        return {
            "intent": "visual_lookup",
            "requires_visual_response": True,
            "image_types": [],
        }

    return {
        "intent": "general_question",
        "requires_visual_response": False,
        "image_types": [],
    }
