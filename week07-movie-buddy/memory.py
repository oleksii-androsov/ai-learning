import datetime
from typing import Optional
import boto3

dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")

profiles_table   = dynamodb.Table("movie_buddy_profiles")
summaries_table  = dynamodb.Table("movie_buddy_summaries")
devices_table    = dynamodb.Table("movie_buddy_devices")


# ---------- Device token → user_id mapping ----------

def get_user_id_for_device(device_token: str) -> Optional[str]:
    resp = devices_table.get_item(Key={"device_token": device_token})
    item = resp.get("Item")
    return item["user_id"] if item else None


def register_device(device_token: str, user_id: str):
    devices_table.put_item(Item={"device_token": device_token, "user_id": user_id})


# ---------- User profile ----------

def get_profile(user_id: str) -> dict:
    resp = profiles_table.get_item(Key={"user_id": user_id})
    return resp.get("Item", {})


def save_profile(user_id: str, profile: dict):
    profile["user_id"] = user_id
    profile["updated_at"] = datetime.date.today().isoformat()
    profiles_table.put_item(Item=profile)


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
    resp = summaries_table.get_item(Key={"user_id": user_id})
    item = resp.get("Item")
    return item["summary"] if item else ""


def save_summary(user_id: str, summary: str):
    summaries_table.put_item(Item={
        "user_id": user_id,
        "summary": summary,
        "updated_at": datetime.date.today().isoformat(),
    })


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
