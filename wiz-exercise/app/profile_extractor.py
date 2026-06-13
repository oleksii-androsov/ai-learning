"""Uses Haiku to extract user facts from a conversation and update the profile."""
import json
import copy
import logging
import datetime
import anthropic

logger = logging.getLogger(__name__)
client = anthropic.Anthropic()

# Keywords that suggest a message may contain profile-worthy information
_EXTRACTION_TRIGGERS = [
    "watch", "seen", "liked", "loved", "love", "hated", "hate", "enjoyed", "enjoy",
    "didn't like", "don't like", "not a fan",
    "movie", "film", "director", "tarantino", "nolan", "spielberg",
    "genre", "comedy", "sci-fi", "fantasy", "horror", "thriller", "drama", "animation",
    "kids", "children", "daughter", "son", "family", "years old",
    "netflix", "prime", "disney", "hbo", "apple tv", "streaming", "subscribe",
    "prefer", "preference", "weather", "cinema", "rain",
    "pixar", "dreamworks", "ghibli", "illumination",
]

EXTRACTION_PROMPT = """You are a profile extraction assistant for Movie Buddy, a film recommendation app.

Read the conversation below and extract any new facts about the user that are worth remembering for future sessions. Only extract things the user explicitly stated — do not infer or guess.

Return a JSON object with only the fields where you found new information. Omit fields where nothing new was learned. Use these fields:

{
  "movies": [{"title": "...", "opinion": "liked|disliked|watched", "notes": "any specific comments"}],
  "genre_preferences": {"liked": ["..."], "disliked": ["..."]},
  "children": [{"stated_age": N}],
  "streaming_platforms": ["..."],
  "weather_preference": "cinema_when_rain|stream_when_rain"
}

For children: store the stated age only. Do not convert to birth year or add dates.
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


def should_extract(user_message: str) -> bool:
    """Quick heuristic — skip Haiku call for short/trivial messages."""
    msg = user_message.lower()
    if len(msg.split()) < 5:
        return False
    result = any(trigger in msg for trigger in _EXTRACTION_TRIGGERS)
    logger.warning(f"should_extract={result} for: {msg[:80]}")
    return result


def extract_profile_updates(conversation: list, existing_profile: dict) -> tuple:
    """Returns (updated_profile, changed: bool). Fails silently on error."""
    try:
        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in conversation
            if isinstance(m.get("content"), str)
        )

        if not conv_text.strip():
            return existing_profile, False

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": conv_text}],
        )

        raw = response.content[0].text.strip()
        logger.info(f"Extractor raw response: {raw[:200]}")

        # Strip markdown code block if present
        import re as _re
        match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, _re.DOTALL)
        text = match.group(1) if match else raw

        if not text:
            return existing_profile, False

        updates = json.loads(text)
    except Exception as e:
        logger.warning(f"Profile extraction failed: {e}")
        return existing_profile, False

    if not updates:
        logger.warning("Extractor returned empty updates — nothing to save")
        return existing_profile, False

    merged = _merge(existing_profile, updates)
    changed = merged != existing_profile
    logger.warning(f"Extraction complete — changed={changed}, updates={list(updates.keys())}")
    return merged, changed


def _merge(existing: dict, updates: dict) -> dict:
    """Merge extracted updates into existing profile. Returns updated profile."""
    if not updates:
        return existing

    merged = copy.deepcopy(existing)

    if "movies" in updates:
        existing_titles = {m["title"].lower() for m in merged.get("movies", [])}
        for m in updates["movies"]:
            if m["title"].lower() not in existing_titles:
                merged.setdefault("movies", []).append(m)
                existing_titles.add(m["title"].lower())

    if "genre_preferences" in updates:
        for key in ("liked", "disliked"):
            new = updates["genre_preferences"].get(key, [])
            existing_list = merged.setdefault("genre_preferences", {}).setdefault(key, [])
            for g in new:
                if g.lower() not in [e.lower() for e in existing_list]:
                    existing_list.append(g)

    if "children" in updates:
        today = datetime.date.today().isoformat()
        merged["children"] = [
            {"stated_age": c["stated_age"], "as_of": today}
            for c in updates["children"]
            if "stated_age" in c and c["stated_age"] is not None
        ]

    if "streaming_platforms" in updates:
        existing_platforms = [p.lower() for p in merged.get("streaming_platforms", [])]
        for p in updates["streaming_platforms"]:
            if p.lower() not in existing_platforms:
                merged.setdefault("streaming_platforms", []).append(p)

    if "weather_preference" in updates:
        merged["weather_preference"] = updates["weather_preference"]

    return merged


def update_summary(user_id: str, conversation: list, prior_summary: str) -> str:
    """Returns updated combined summary string. Fails silently on error."""
    try:
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
    except Exception as e:
        logger.warning(f"Summary update failed: {e}")
        return prior_summary
