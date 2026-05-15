import datetime
import logging
from typing import Optional
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

try:
    dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")
    profiles_table  = dynamodb.Table("movie_buddy_profiles")
    summaries_table = dynamodb.Table("movie_buddy_summaries")
    devices_table   = dynamodb.Table("movie_buddy_devices")
    _DYNAMO_AVAILABLE = True
except Exception:
    _DYNAMO_AVAILABLE = False
    logger.warning("DynamoDB unavailable — memory features disabled")


# ---------- Device token → user_id mapping ----------

def get_user_id_for_device(device_token: str) -> Optional[str]:
    if not _DYNAMO_AVAILABLE:
        return None
    try:
        resp = devices_table.get_item(Key={"device_token": device_token})
        item = resp.get("Item")
        return item["user_id"] if item else None
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"DynamoDB get_user_id_for_device failed: {e}")
        return None


def register_device(device_token: str, user_id: str):
    if not _DYNAMO_AVAILABLE:
        return
    try:
        devices_table.put_item(Item={"device_token": device_token, "user_id": user_id})
    except Exception as e:
        logger.warning(f"DynamoDB register_device failed: {e}")


# ---------- User profile ----------

def get_profile(user_id: str) -> dict:
    if not _DYNAMO_AVAILABLE:
        return {}
    try:
        resp = profiles_table.get_item(Key={"user_id": user_id})
        return resp.get("Item", {})
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"DynamoDB get_profile failed: {e}")
        return {}


def save_profile(user_id: str, profile: dict):
    if not _DYNAMO_AVAILABLE:
        return
    try:
        profile["user_id"] = user_id
        profile["updated_at"] = datetime.date.today().isoformat()
        # DynamoDB rejects None values — strip them before writing
        clean = {k: v for k, v in profile.items() if v is not None}
        profiles_table.put_item(Item=clean)
    except Exception as e:
        logger.warning(f"DynamoDB save_profile failed: {e}")


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
    if not _DYNAMO_AVAILABLE:
        return ""
    try:
        resp = summaries_table.get_item(Key={"user_id": user_id})
        item = resp.get("Item")
        return item["summary"] if item else ""
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"DynamoDB get_summary failed: {e}")
        return ""


def save_summary(user_id: str, summary: str):
    if not _DYNAMO_AVAILABLE:
        return
    try:
        summaries_table.put_item(Item={
            "user_id": user_id,
            "summary": summary,
            "updated_at": datetime.date.today().isoformat(),
        })
    except Exception as e:
        logger.warning(f"DynamoDB save_summary failed: {e}")


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
        current_year = datetime.date.today().year
        ages = [current_year - c["birth_year"] for c in children]
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
