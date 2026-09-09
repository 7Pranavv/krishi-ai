"""Season detection and crop-season suitability.

India's cropping calendar shifts by region: the southwest monsoon reaches the
south and east earlier and lingers, so Kharif runs longer there than in the
north. Ported from AhqafCoder/AICropRecommendation (MIT), with the state-to-
region mapping added so it works from the states our rainfall model already
knows.

This is a lookup, not a model - it never claims more precision than a calendar.
"""
from __future__ import annotations

from datetime import date

# Months belonging to each season, by region.
SEASON_MONTHS: dict[str, dict[str, list[int]]] = {
    "north_india": {"kharif": [6, 7, 8, 9, 10], "rabi": [11, 12, 1, 2, 3], "zaid": [4, 5]},
    "south_india": {"kharif": [6, 7, 8, 9, 10, 11], "rabi": [12, 1, 2, 3, 4], "zaid": [5]},
    "west_india":  {"kharif": [6, 7, 8, 9, 10], "rabi": [11, 12, 1, 2, 3], "zaid": [4, 5]},
    "east_india":  {"kharif": [6, 7, 8, 9, 10, 11], "rabi": [12, 1, 2, 3], "zaid": [4, 5]},
    "default":     {"kharif": [6, 7, 8, 9, 10], "rabi": [11, 12, 1, 2, 3], "zaid": [4, 5]},
}

# The 20 states the rainfall model knows, plus the common remainder.
STATE_REGION: dict[str, str] = {
    "Punjab": "north_india", "Haryana": "north_india", "Uttar Pradesh": "north_india",
    "Uttarakhand": "north_india", "Himachal Pradesh": "north_india",
    "Delhi": "north_india", "Jammu and Kashmir": "north_india",
    "Rajasthan": "west_india", "Gujarat": "west_india", "Maharashtra": "west_india",
    "Madhya Pradesh": "west_india", "Goa": "west_india",
    "Karnataka": "south_india", "Kerala": "south_india", "Tamil Nadu": "south_india",
    "Andhra Pradesh": "south_india", "Telangana": "south_india",
    "West Bengal": "east_india", "Bihar": "east_india", "Odisha": "east_india",
    "Jharkhand": "east_india", "Assam": "east_india", "Chhattisgarh": "east_india",
    # The mandi price feed spells several states differently from the rainfall
    # model. Without these a Kerala farmer silently got the generic calendar,
    # which ends Kharif a month early for the south.
    "Keralam": "south_india",
    "Chattisgarh": "east_india",
    "NCT of Delhi": "north_india",
    "Manipur": "east_india",
    "Meghalaya": "east_india", "Tripura": "east_india", "Nagaland": "east_india",
    "Mizoram": "east_india", "Arunachal Pradesh": "east_india", "Sikkim": "east_india",
    "Ladakh": "north_india",
    "Puducherry": "south_india", "Andaman and Nicobar Islands": "south_india",
}

REGION_LABELS = {
    "north_india": "North India", "south_india": "South India",
    "west_india": "West India", "east_india": "East India",
    "default": "India (general)",
}

SEASON_LABELS = {
    "kharif": "Kharif (monsoon)",
    "rabi": "Rabi (winter)",
    "zaid": "Zaid (summer)",
}

SEASON_NOTES = {
    "kharif": "Sown with the monsoon, harvested after it. Rain-fed crops do well now.",
    "rabi": "Sown after the monsoon into stored soil moisture, harvested in spring. "
            "Usually needs irrigation.",
    "zaid": "Short summer season between harvests. Needs assured irrigation and "
            "heat-tolerant crops.",
}

# Which crops fit which season. This is general agronomic guidance for the
# season, NOT a list of what the recommender can score - it deliberately
# includes staples like wheat and potato that the crop model does not cover,
# because they are the right answer for the season and a farmer should see
# them. The UI labels this section accordingly.
SEASON_CROPS: dict[str, dict[str, list[str]]] = {
    "kharif": {
        "highly_suitable": ["rice", "maize", "cotton", "jute", "pigeonpeas",
                            "mothbeans", "mungbean", "blackgram", "soybean",
                            "groundnut", "sorghum", "bajra"],
        "suitable": ["sesame", "sunflower", "kidneybeans", "banana", "sugarcane"],
        "not_suitable": ["wheat", "barley", "chickpea", "lentil", "mustard"],
    },
    "rabi": {
        "highly_suitable": ["wheat", "barley", "chickpea", "lentil", "mustard", "potato"],
        "suitable": ["onion", "cabbage", "sunflower", "maize", "tomato"],
        "not_suitable": ["rice", "cotton", "jute", "mothbeans"],
    },
    "zaid": {
        "highly_suitable": ["watermelon", "muskmelon", "mungbean", "cucumber"],
        "suitable": ["maize", "sesame", "groundnut", "tomato"],
        "not_suitable": ["wheat", "rice", "cotton", "sugarcane", "chickpea"],
    },
}


def region_for_state(state: str | None) -> str:
    return STATE_REGION.get((state or "").strip(), "default")


def detect(month: int | None = None, state: str | None = None) -> dict:
    """Which season a month falls in for a state's region."""
    month = month or date.today().month
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")

    region = region_for_state(state)
    months = SEASON_MONTHS[region]
    season = next((s for s, ms in months.items() if month in ms), "kharif")

    return {
        "season": season,
        "season_label": SEASON_LABELS[season],
        "region": region,
        "region_label": REGION_LABELS[region],
        "month": month,
        "note": SEASON_NOTES[season],
        "months_in_season": sorted(months[season]),
        "crops": SEASON_CROPS[season],
    }


def suitability(crop: str, season: str) -> str | None:
    """How well a crop fits a season: highly_suitable / suitable / not_suitable.

    None when the crop is not classified for that season - the caller should
    say nothing rather than guess.
    """
    table = SEASON_CROPS.get(season)
    if not table:
        return None
    crop = crop.lower()
    for level, crops in table.items():
        if crop in crops:
            return level
    return None
