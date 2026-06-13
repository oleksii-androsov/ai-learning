import datetime
import logging
import os
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

try:
    _client = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=3000)
    _db = _client["movie_buddy"]
    profiles_col  = _db["profiles"]
    summaries_col = _db["summaries"]
    devices_col   = _db["devices"]
    _MONGO_AVAILABLE = True
except Exception:
    _MONGO_AVAILABLE = False
    logger.warning("MongoDB unavailable — memory features disabled")


# ---------- Device token → user_id mapping ----------

def get_user_id_for_device(device_token: str) -> Optional[str]:
    if not _MONGO_AVAILABLE:
        return None
    try:
        item = devices_col.find_one({"device_token": device_token})
        return item["user_id"] if item else None
    except PyMongoError as e:
        logger.warning(f"MongoDB get_user_id_for_device failed: {e}")
        return None


def register_device(device_token: str, user_id: str):
    if not _MONGO_AVAILABLE:
        return
    try:
        devices_col.replace_one(
            {"device_token": device_token},
            {"device_token": device_token, "user_id": user_id},
            upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"MongoDB register_device failed: {e}")


# ---------- User profile ----------

def get_profile(user_id: str) -> dict:
    if not _MONGO_AVAILABLE:
        return {}
    try:
        item = profiles_col.find_one({"user_id": user_id})
        if item:
            item.pop("_id", None)
        return item or {}
    except PyMongoError as e:
        logger.warning(f"MongoDB get_profile failed: {e}")
        return {}


def save_profile(user_id: str, profile: dict):
    if not _MONGO_AVAILABLE:
        return
    try:
        profile["user_id"] = user_id
        profile["updated_at"] = datetime.date.today().isoformat()
        profiles_col.replace_one({"user_id": user_id}, profile, upsert=True)
    except PyMongoError as e:
        logger.warning(f"MongoDB save_profile failed: {e}")


def empty_profile(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "movies": [],
        "genre_preferences": {"liked": [], "disliked": []},
        "children": [],
        "streaming_platforms": [],
        "weather_preference": None,
        "updated_at": datetime.date.today().isoformat(),
    }


# ---------- Conversation summaries ----------

def get_summary(user_id: str) -> str:
    if not _MONGO_AVAILABLE:
        return ""
    try:
        item = summaries_col.find_one({"user_id": user_id})
        return item["summary"] if item else ""
    except PyMongoError as e:
        logger.warning(f"MongoDB get_summary failed: {e}")
        return ""


def save_summary(user_id: str, summary: str):
    if not _MONGO_AVAILABLE:
        return
    try:
        summaries_col.replace_one(
            {"user_id": user_id},
            {"user_id": user_id, "summary": summary, "updated_at": datetime.date.today().isoformat()},
            upsert=True
        )
    except PyMongoError as e:
        logger.warning(f"MongoDB save_summary failed: {e}")


# ---------- Profile formatting for Orchestrator prompt ----------

def format_profile_for_prompt(profile: dict) -> str:
    if not profile:
        return ""

    lines = ["## What I know about this user"]

    if profile.get("movies"):
        lines.append("\nMovies they've watched:")
        for m in profile["movies"]:
            line = f"  - {m['title']}: {m.get('opinion', 'watched')}"
            if m.get("notes"):
                line += f" ({m['notes']})"
            lines.append(line)

    liked = profile.get("genre_preferences", {}).get("liked", [])
    disliked = profile.get("genre_preferences", {}).get("disliked", [])
    if liked:
        lines.append(f"\nGenres they enjoy: {', '.join(liked)}")
    if disliked:
        lines.append(f"Genres they tend to avoid: {', '.join(disliked)}")

    children = profile.get("children", [])
    if children:
        today = datetime.date.today()
        ages = []
        for c in children:
            if "stated_age" in c and "as_of" in c and c["stated_age"] is not None:
                as_of = datetime.date.fromisoformat(str(c["as_of"]))
                years_passed = (today - as_of).days // 365
                ages.append(int(c["stated_age"]) + years_passed)
            elif "birth_year" in c:
                ages.append(today.year - int(c["birth_year"]))
        if ages:
            lines.append(f"\nHas children aged: {', '.join(str(a) for a in ages)}")

    platforms = profile.get("streaming_platforms", [])
    if platforms:
        lines.append(f"\nStreaming platforms: {', '.join(platforms)}")

    weather_pref = profile.get("weather_preference")
    if weather_pref == "cinema_when_rain":
        lines.append("\nWeather preference: enjoys going to the cinema even in rainy weather")
    elif weather_pref == "stream_when_rain":
        lines.append("\nWeather preference: prefers streaming at home when it rains, cinema only on nice days")

    return "\n".join(lines)


def format_summary_for_prompt(summary: str) -> str:
    if not summary:
        return ""
    return f"## Previous conversations\n{summary}"
