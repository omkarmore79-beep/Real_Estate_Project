def simple_query(data, question):
    question = question.lower()

    for item in data:
        if "price" in question:
            return item.get("price_details")

        if "amenities" in question:
            return item.get("amenities")

    return "No relevant data found"