"""Agronomic reference tables.

These are lifted verbatim from the original Streamlit apps (ml-models/*/app.py)
so the integrated product keeps the same advice the standalone tools gave.
Nothing here is generated - if a crop is missing from a table the API returns
null rather than inventing text.
"""

# ---------------------------------------------------------------- crop
CROP_INFO: dict[str, dict[str, str]] = {
    "rice":        {"emoji": "\U0001F33E", "season": "Kharif", "tip": "Needs flooded fields and high humidity."},
    "maize":       {"emoji": "\U0001F33D", "season": "Kharif", "tip": "Grows well in well-drained loamy soil."},
    "chickpea":    {"emoji": "\U0001FAD8", "season": "Rabi",   "tip": "Drought tolerant, needs cool dry weather."},
    "kidneybeans": {"emoji": "\U0001FAD8", "season": "Kharif", "tip": "Needs well-drained fertile soil."},
    "pigeonpeas":  {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Drought resistant, good for dry regions."},
    "mothbeans":   {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Thrives in arid and semi-arid regions."},
    "mungbean":    {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Short duration crop, good for soil health."},
    "blackgram":   {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Grows in tropical and subtropical regions."},
    "lentil":      {"emoji": "\U0001FAD8", "season": "Rabi",   "tip": "Cool season crop, needs well-drained soil."},
    "pomegranate": {"emoji": "\U0001F34E", "season": "Annual", "tip": "Drought tolerant, needs hot dry summers."},
    "banana":      {"emoji": "\U0001F34C", "season": "Annual", "tip": "Needs high humidity and rich soil."},
    "mango":       {"emoji": "\U0001F96D", "season": "Annual", "tip": "Tropical fruit, needs dry flowering season."},
    "grapes":      {"emoji": "\U0001F347", "season": "Annual", "tip": "Needs well-drained soil and dry summers."},
    "watermelon":  {"emoji": "\U0001F349", "season": "Summer", "tip": "Needs sandy loam soil and warm weather."},
    "muskmelon":   {"emoji": "\U0001F348", "season": "Summer", "tip": "Grows in warm climate with sandy soil."},
    "apple":       {"emoji": "\U0001F34E", "season": "Annual", "tip": "Needs cold winters for proper fruiting."},
    "orange":      {"emoji": "\U0001F34A", "season": "Annual", "tip": "Subtropical fruit, needs well-drained soil."},
    "papaya":      {"emoji": "\U0001F348", "season": "Annual", "tip": "Fast growing, needs warm humid climate."},
    "coconut":     {"emoji": "\U0001F965", "season": "Annual", "tip": "Coastal crop, needs high humidity."},
    "cotton":      {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Needs black soil and moderate rainfall."},
    "jute":        {"emoji": "\U0001F33F", "season": "Kharif", "tip": "Needs high humidity and alluvial soil."},
    "coffee":      {"emoji": "☕",     "season": "Annual", "tip": "Grows in hilly regions with shade."},
    # Added when the model was retrained to 26 crops - see training/train_crop.py.
    "arecanut":    {"emoji": "\U0001F334", "season": "Annual", "tip": "Coastal palm. Needs humid air, partial shade and steady irrigation."},
    "jackfruit":   {"emoji": "\U0001F333", "season": "Annual", "tip": "Hardy tree crop. Needs deep, well-drained soil and little care once established."},
    "sugarcane":   {"emoji": "\U0001F33E", "season": "Annual", "tip": "Long crop of 12-18 months. Needs rich soil and heavy, reliable irrigation."},
    "groundnut":   {"emoji": "\U0001F95C", "season": "Kharif", "tip": "Light, well-drained soil suits it. Apply gypsum at pegging for good pod filling."},
}

# ---------------------------------------------------------------- fertilizer
SOIL_LABELS = {
    "black_cotton": "Black Cotton", "clay": "Clay", "clay_loam": "Clay Loam",
    "loamy": "Loamy", "red_laterite": "Red Laterite", "sandy": "Sandy",
    "sandy_loam": "Sandy Loam",
}
IRRIGATION_LABELS = {
    "drip": "Drip / Fertigation",
    "irrigated": "Canal / Borewell Irrigated",
    "rainfed": "Rainfed",
}
PREV_CROP_LABELS = {
    "cereal": "Cereal (Wheat/Rice/Maize)", "fallow": "Fallow (No crop)",
    "legume": "Legume (Pulse/Soybean)", "sugarcane": "Sugarcane",
    "vegetable": "Vegetable",
}
GROWTH_STAGE_LABELS = {
    "initial": "Initial", "development": "Development",
    "mid": "Mid Season", "late": "Late Season",
}

FERTILIZER_SCHEDULES: dict[str, list[tuple[str, str]]] = {
    "cereal": [
        ("Basal (At sowing)", "Apply full P and K + 1/3 N as DAP/MOP before sowing. Mix into top 10 cm soil."),
        ("First top-dress (21 DAP)", "Apply 1/3 N as Urea when crop is 3 weeks old. Irrigate immediately after."),
        ("Second top-dress (45 DAP)", "Apply remaining 1/3 N at tillering/panicle initiation stage."),
    ],
    "pulse": [
        ("Basal (At sowing)", "Apply full P and K + starter N dose. Pulses fix their own N - avoid excess N."),
        ("Foliar spray (30 DAP)", "Apply 2% DAP foliar spray at flowering to boost pod set."),
    ],
    "oilseed": [
        ("Basal (At sowing)", "Apply full P and K + 1/2 N before sowing."),
        ("Top-dress (30 DAP)", "Apply remaining 1/2 N at branching stage."),
        ("Foliar (Flowering)", "Apply 0.5% Borax spray at flowering to improve seed set."),
    ],
    "vegetable": [
        ("Basal (Land prep)", "Apply full P and K + 1/3 N with FYM during field preparation."),
        ("First top-dress (15 DAP)", "Apply 1/3 N after transplanting/thinning."),
        ("Second top-dress (35 DAP)", "Apply remaining 1/3 N at fruit development stage."),
    ],
    "cash_crop": [
        ("Basal (At planting)", "Apply full P and K + 1/4 N at planting."),
        ("First ratoon (60 DAP)", "Apply 1/4 N at 60 days."),
        ("Second ratoon (120 DAP)", "Apply 1/4 N at 120 days."),
        ("Final dose (180 DAP)", "Apply remaining 1/4 N at grand growth phase."),
    ],
}

CROP_SCHEDULE_MAP = {
    "rice": "cereal", "wheat": "cereal", "maize": "cereal", "sorghum": "cereal",
    "bajra": "cereal", "barley": "cereal",
    "chickpea": "pulse", "lentil": "pulse", "pigeonpeas": "pulse",
    "mungbean": "pulse", "soybean": "pulse",
    "groundnut": "oilseed", "mustard": "oilseed", "sunflower": "oilseed",
    "sesame": "oilseed",
    "potato": "vegetable", "tomato": "vegetable", "onion": "vegetable",
    "cabbage": "vegetable",
    "sugarcane": "cash_crop", "cotton": "cash_crop", "jute": "cash_crop",
    "banana": "cash_crop", "mango": "cash_crop",
}

FERTILIZER_TIPS = {
    "rice": "Split N application is critical for rice. Never apply all N at once - it causes lodging and increases blast risk.",
    "wheat": "Apply Zinc Sulphate (25 kg/ha) as basal if soil Zn is deficient. Common in Indo-Gangetic plains.",
    "maize": "Maize is a heavy feeder. Ensure adequate K - deficiency shows as yellowing of leaf margins.",
    "sugarcane": "Trash mulching after harvest reduces K requirement by 30%. Use trash as organic mulch.",
    "cotton": "Avoid excess N in cotton - it promotes vegetative growth at the cost of boll formation.",
    "potato": "Potatoes need high K for tuber quality. Deficiency causes hollow heart and poor storability.",
    "tomato": "Use calcium nitrate as part of N source to prevent blossom end rot.",
    "chickpea": "Rhizobium inoculation of seeds can reduce N fertilizer need by 50% in chickpea.",
    "soybean": "Soybean fixes 60-80 kg N/ha through symbiosis. Starter N of 20-25 kg/ha is sufficient.",
    "mustard": "Sulphur (20-30 kg/ha as gypsum) is critical for mustard oil quality and yield.",
    "groundnut": "Gypsum application (500 kg/ha) at pegging stage is essential for pod filling in groundnut.",
    "banana": "Banana responds well to fertigation. Split into 12 monthly doses for best results.",
    "onion": "Excess N in onion causes thick necks and poor storage. Reduce N in last 30 days.",
    "default": "Always do a soil test before applying fertilizers. Over-application wastes money and harms soil health.",
}

# Nutrient content of the commercial products, used to convert kg/ha of
# nutrient into kg/ha of bagged fertilizer (same factors as the Streamlit app).
COMMERCIAL_CONTENT = {"urea": 0.46, "dap": 0.46, "mop": 0.60}

# ---------------------------------------------------------------- water
WATER_CROP_TIPS = {
    "rice": "Rice needs flooded conditions. Maintain 5-10 cm standing water during mid-season.",
    "wheat": "Wheat is sensitive to water stress at crown root initiation and grain filling stages.",
    "maize": "Critical water periods: tasseling and silking. Avoid stress during these stages.",
    "sugarcane": "Sugarcane needs consistent moisture. Drip irrigation can save up to 40% water.",
    "cotton": "Reduce irrigation after boll opening. Excess water causes boll rot.",
    "soybean": "Most sensitive to drought during pod filling. Maintain soil moisture at 50-70% FC.",
    "potato": "Tuber initiation and bulking are critical stages. Avoid waterlogging.",
    "tomato": "Drip irrigation recommended. Inconsistent watering causes blossom end rot.",
    "onion": "Reduce irrigation 2 weeks before harvest to improve storage quality.",
    "chickpea": "Drought tolerant. One irrigation at flowering significantly boosts yield.",
    "groundnut": "Critical stages: pegging and pod development. Avoid waterlogging.",
    "sunflower": "Deep-rooted crop. Irrigate at head formation and seed filling stages.",
    "banana": "High water demand. Drip or micro-sprinkler irrigation is most efficient.",
    "mango": "Withhold irrigation during flowering to improve fruit set.",
    "mustard": "Sensitive at flowering. One well-timed irrigation can increase yield by 30%.",
}

# Field-area units accepted by the water endpoint, expressed in hectares.
AREA_UNITS = {"hectare": 1.0, "acre": 0.404686, "bigha": 0.2529}

# ---------------------------------------------------------------- rainfall
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

MONTH_SEASON = {
    1: "Winter", 2: "Winter", 3: "Pre-Kharif", 4: "Pre-Kharif",
    5: "Pre-Kharif", 6: "Kharif", 7: "Kharif", 8: "Kharif",
    9: "Kharif", 10: "Rabi", 11: "Rabi", 12: "Rabi",
}

RAINFALL_ADVICE = {
    "Deficient": {
        "Kharif": "Low rainfall expected during Kharif. Consider drought-tolerant crops like bajra, jowar, or moong. Plan for supplemental irrigation.",
        "Rabi": "Dry Rabi season ahead. Wheat and mustard may need extra irrigation. Check groundwater levels before sowing.",
        "Winter": "Dry winter conditions. Protect rabi crops from moisture stress. Mulching can help retain soil moisture.",
        "Pre-Kharif": "Deficient pre-monsoon rains. Delay sowing until adequate moisture is available. Prepare water harvesting structures.",
    },
    "Normal": {
        "Kharif": "Normal monsoon expected. Good conditions for rice, maize, soybean, and cotton. Proceed with planned sowing schedule.",
        "Rabi": "Adequate moisture for Rabi crops. Wheat, chickpea, and mustard should perform well. Monitor for fungal diseases.",
        "Winter": "Normal winter conditions. Good for vegetable cultivation and rabi crop establishment.",
        "Pre-Kharif": "Normal pre-monsoon showers. Prepare fields for Kharif sowing. Good time for land preparation.",
    },
    "Excess": {
        "Kharif": "Heavy rainfall expected. Risk of waterlogging and flooding. Ensure proper field drainage. Avoid low-lying areas for sowing.",
        "Rabi": "Excess moisture may delay Rabi sowing. Watch for root rot and fungal diseases. Improve field drainage.",
        "Winter": "Heavy winter rains. Protect crops from lodging. Delay fertilizer application until fields dry.",
        "Pre-Kharif": "Heavy pre-monsoon rains. Good for soil moisture recharge but risk of soil erosion. Use contour bunding.",
    },
}


def soil_nutrient_status(value: float, low: float, high: float) -> str:
    """ICAR soil-test bands used by the fertilizer tool."""
    if value < low:
        return "Low"
    if value < high:
        return "Medium"
    return "High"


def prettify_disease_label(label: str) -> tuple[str, str, bool]:
    """Split a PlantVillage class name into (crop, condition, is_healthy).

    "Tomato___Late_blight" becomes ("Tomato", "Late blight", False).
    Only reformats what the label already contains - nothing is inferred.
    """
    crop, _, condition = label.partition("___")
    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip() or "Unknown"
    healthy = condition.lower() == "healthy"
    # Labels arrive inconsistently cased ("Early_blight" but "healthy"), and
    # the condition is the page's main heading, so normalise the first letter.
    condition = condition[:1].upper() + condition[1:]
    return crop, condition, healthy
