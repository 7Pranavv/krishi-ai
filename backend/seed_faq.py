"""Seed the curated FAQ table.

These answers are served before the AI is consulted: instant, free, and
identical every time. Carried over from the Shoora chatbot's ChatBotModel idea.

Run:  python seed_faq.py
"""
from sqlmodel import Session, select

from app.db import engine, init_db
from app.models import FaqEntry

# Questions are matched case-insensitively on the exact text, so keep the keys
# lowercase and phrase them the way a farmer would type them.
ENTRIES: list[tuple[str, str]] = [
    (
        "what is krishi.ai",
        "Krishi.AI is a free digital assistant for farmers. You can ask farming "
        "questions in your own language, find the right crop for your soil, get "
        "fertilizer and irrigation plans, check the rainfall outlook for your "
        "state, and photograph a leaf to check it for disease.",
    ),
    (
        "what can you do",
        "I can answer farming questions, and Krishi.AI has five tools: Crop "
        "Recommendation, Plant Disease Detection, Fertilizer Plan, Water "
        "Management and Rainfall Forecast. You will find all of them on the home page.",
    ),
    (
        "how do i check my plant for disease",
        "Open Plant Disease Detection from the menu, take a clear photo of one "
        "affected leaf against a plain background, and upload it. You will get the "
        "likely disease with a confidence score.",
    ),
    (
        "which crop should i grow",
        "Open Crop Recommendation and enter your soil nitrogen, phosphorus and "
        "potassium values along with your local temperature, humidity, pH and "
        "rainfall. If you do not have a soil test, your nearest Krishi Vigyan "
        "Kendra can do one for you.",
    ),
    (
        "how much water does my crop need",
        "Open Water Management, choose your crop, soil type and growth stage, and "
        "enter today's weather. You will get the millimetres of water needed per "
        "day, the total litres for your field, and how many days to leave between "
        "irrigations.",
    ),
]


def main() -> None:
    init_db()
    added = 0
    with Session(engine) as session:
        for question, answer in ENTRIES:
            existing = session.exec(
                select(FaqEntry).where(FaqEntry.question == question)
            ).first()
            if existing:
                existing.answer = answer  # keep the wording current
                session.add(existing)
            else:
                session.add(FaqEntry(question=question, answer=answer))
                added += 1
        session.commit()
    print(f"FAQ seeded: {added} new, {len(ENTRIES) - added} updated.")


if __name__ == "__main__":
    main()
