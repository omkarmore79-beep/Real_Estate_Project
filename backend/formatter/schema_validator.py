from pydantic import BaseModel
from typing import List


class RealEstateSchema(BaseModel):
    project_name: str
    buildings: List[str]
    developer: str
    location: str
    property_type: str
    configurations: List[str]
    price_details: List[str]
    amenities: List[str]
    floor_plans: List[str]
    possession_date: str
    contact_details: List[str]
