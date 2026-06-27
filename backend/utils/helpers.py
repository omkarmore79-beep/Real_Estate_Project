def safe_json_load(text):
    import json

    try:
        return json.loads(text)
    except:
        return None