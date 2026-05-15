"""Uses Haiku to extract user facts from a conversation and update the profile."""
import json
import datetime
import anthropic

client = anthropic.Anthropic()

EXTRACTION_PROMPT = """You are a profile extraction assistant for Movie Buddy, a film recommendation app.

Read the conversation below and extract any new facts about the user that are worth remembering for future sessions. Only extract things the user explicitly stated — do not infer or guess.

Return a JSON object with only the fields where you found new information. Omit fields where nothing new was learned. Use these fields:

{
  "movies": [{"title": "...", "opinion": "liked|disliked|watched", "notes": "any specific comments"}],
  "genre_preferences": {"liked": ["..."], "disliked": ["..."]},
  "children": [{"birth_year": YYYY}],
  "streaming_platforms": ["..."],
  "weather_preference": "cinema_when_rain|stream_when_rain"
}

For children: if the user mentions ages, convert to birth year using today's year.
For weather_preference: only set if the user explicitly stated a preference.
If nothing new was learned, return an empty object: {}

Return only valid JSON, no explanation."""


SUMMARY_PROMPT = """You are summarising a Movie Buddy conversation for long-term memory.

Write a 2-4 sentence summary of what was discussed: what the user was looking for, what was recommended, and any relevant context. Be specific about film titles and preferences mentioned. Write in past tense.

Previous summary of all prior sessions (may be empty):
{prior_summary}

New conversation to summarise:
{conversation}

Return only the updated combined summary — a single paragraph merging prior history with today's session."""


def extract_profile_updates(conversation: list[dict], existing_profile: dict) -> dict:
    """Returns a dict of fields to merge into the profile. Empty dict = nothing new."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation
        if isinstance(m.get("content"), str)
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": conv_text}],
    )

    try:
        updates = json.loads(response.content[0].text.strip())
    except (json.JSONDecodeError, IndexError):
        return {}

    return _merge(existing_profile, updates)


def _merge(existing: dict, updates: dict) -> dict:
    """Merge extracted updates into existing profile. Returns updated profile."""
    if not updates:
        return existing

    merged = dict(existing)

    # Movies — add new ones, skip duplicates by title
    if "movies" in updates:
        existing_titles = {m["title"].lower() for m in merged.get("movies", [])}
        for m in updates["movies"]:
            if m["title"].lower() not in existing_titles:
                merged.setdefault("movies", []).append(m)
                existing_titles.add(m["title"].lower())

    # Genre preferences — union, no duplicates
    if "genre_preferences" in updates:
        for key in ("liked", "disliked"):
            new = updates["genre_preferences"].get(key, [])
            existing_list = merged.setdefault("genre_preferences", {}).setdefault(key, [])
            for g in new:
                if g.lower() not in [e.lower() for e in existing_list]:
                    existing_list.append(g)

    # Children — replace entirely if new info provided
    if "children" in updates:
        merged["children"] = updates["children"]

    # Streaming platforms — union
    if "streaming_platforms" in updates:
        existing_platforms = [p.lower() for p in merged.get("streaming_platforms", [])]
        for p in updates["streaming_platforms"]:
            if p.lower() not in existing_platforms:
                merged.setdefault("streaming_platforms", []).append(p)

    # Weather preference — overwrite if explicitly stated
    if "weather_preference" in updates:
        merged["weather_preference"] = updates["weather_preference"]

    return merged


def update_summary(user_id: str, conversation: list[dict], prior_summary: str) -> str:
    """Returns updated combined summary string."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation
        if isinstance(m.get("content"), str)
    )

    prompt = SUMMARY_PROMPT.format(
        prior_summary=prior_summary or "(none)",
        conversation=conv_text,
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()
