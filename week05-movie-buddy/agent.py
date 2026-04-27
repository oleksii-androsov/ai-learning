import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import anthropic
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
TMDB_API_KEY = os.environ["TMDB_API_KEY"]
TMDB_BASE = "https://api.themoviedb.org/3"

TMDB_GENRES = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "documentary": 99, "drama": 18, "family": 10751, "fantasy": 14,
    "horror": 27, "music": 10402, "mystery": 9648, "romance": 10749,
    "science fiction": 878, "sci-fi": 878, "thriller": 53, "western": 37,
}


client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

tools = [
    {
        "name": "discover_movies",
        "description": "Search for movie recommendations based on mood, genre, and who is watching. Use this when the user wants suggestions but doesn't have a specific title in mind.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "description": "The mood or tone the user is looking for, e.g. 'light and funny', 'tense thriller', 'emotional drama'"
                },
                "genre": {
                    "type": "string",
                    "description": "The genre the user prefers, e.g. 'sci-fi', 'animation', 'horror'"
                },
                "who_is_watching": {
                    "type": "string",
                    "description": "Who will be watching, e.g. 'solo', 'couple', 'family with young kids', 'group of friends'"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_current_listings",
        "description": "Find movies currently in theaters or recently added to streaming services. For streaming, uses country-specific data so results are accurate for the user's region. Use this when the user wants to know what's available right now. If the user asks for more options, increment the page number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code, e.g. 'DE' for Germany, 'GB' for UK, 'US' for USA. Infer from the user's location."
                },
                "format": {
                    "type": "string",
                    "description": "Where to watch: 'theaters', 'streaming', or 'both'",
                    "enum": ["theaters", "streaming", "both"]
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for results, default 1. Use 2, 3 etc. when the user asks for more options."
                },
                "genre": {
                    "type": "string",
                    "description": "Optional genre filter, e.g. 'documentary', 'animation', 'comedy', 'sci-fi'. Use this when the user asks for a specific genre of current or upcoming content."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_upcoming_listings",
        "description": "Find movies coming soon to theaters or streaming. Use this when the user wants to plan ahead. For streaming, returns country-accurate results. If the user asks for more options, increment the page number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code, e.g. 'DE' for Germany, 'GB' for UK. Infer from the user's location."
                },
                "weeks_ahead": {
                    "type": "integer",
                    "description": "How many weeks ahead to look, e.g. 2 for the next two weeks"
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for results, default 1. Use 2, 3 etc. when the user asks for more options."
                }
            },
            "required": []
        }
    },
    {
        "name": "find_similar",
        "description": "Find movies similar to one the user already likes. If no aspect is specified, uses TMDB recommendations based on genre, cast, and themes. If the user specifies a creative aspect — e.g. 'I loved the music', 'same cinematographer', 'same animator' — use that aspect to search instead, as TMDB won't have that data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The movie the user liked, e.g. 'Dune: Part One'"
                },
                "aspect": {
                    "type": "string",
                    "description": "Optional: a specific creative angle to match on, e.g. 'Hans Zimmer soundtrack', 'Andreas Deja animation', 'Roger Deakins cinematography'. When provided, a web search is used instead of TMDB."
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "get_showtimes",
        "description": "Look up screening times for a specific movie at a specific cinema. Use this when the user asks about showtimes at a named cinema.",
        "input_schema": {
            "type": "object",
            "properties": {
                "movie": {
                    "type": "string",
                    "description": "The movie title to look up showtimes for"
                },
                "cinema": {
                    "type": "string",
                    "description": "The name of the cinema, e.g. 'Astor Film Lounge MyZeil'"
                },
                "city": {
                    "type": "string",
                    "description": "The city where the cinema is located"
                },
                "date": {
                    "type": "string",
                    "description": "Optional: specific date to check, e.g. 'Saturday May 10 2026'. If not provided, returns current or nearest available schedule."
                }
            },
            "required": ["movie", "cinema", "city"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the weather forecast for a city. Use this to decide whether to recommend a theater visit or a stay-at-home streaming option. If the weather is bad, lean toward streaming suggestions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the forecast for, e.g. 'Frankfurt', 'London'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_movie_details",
        "description": "Look up structured details about a specific movie by title: rating, runtime, genres, director, top cast, plot summary, and TMDB score. Use this when the user mentions a specific film and you need accurate facts about it. If the title is ambiguous or part of a franchise with multiple entries (e.g. 'Avengers', 'Dune', 'Lord of the Rings'), ask the user to clarify which one they mean before calling this tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The movie title to look up, e.g. 'Arrival', 'Dune: Part Two'"
                }
            },
            "required": ["title"]
        }
    }
]


def get_movie_details(title):
    search = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"api_key": TMDB_API_KEY, "query": title}
    ).json()

    if "results" not in search:
        return f"TMDB error: {search.get('status_message', search)}"

    if not search["results"]:
        return f"No results found for '{title}'"

    movie_id = search["results"][0]["id"]

    details = requests.get(
        f"{TMDB_BASE}/movie/{movie_id}",
        params={"api_key": TMDB_API_KEY, "append_to_response": "credits"}
    ).json()

    director = next(
        (c["name"] for c in details["credits"]["crew"] if c["job"] == "Director"),
        "Unknown"
    )
    cast = ", ".join(m["name"] for m in details["credits"]["cast"][:5])
    genres = ", ".join(g["name"] for g in details["genres"])

    return (
        f"Title: {details['title']} ({details.get('release_date', '')[:4]})\n"
        f"Rating: {details['vote_average']:.1f}/10 ({details['vote_count']} votes)\n"
        f"Runtime: {details.get('runtime', 'N/A')} min\n"
        f"Genres: {genres}\n"
        f"Director: {director}\n"
        f"Cast: {cast}\n"
        f"Overview: {details['overview']}"
    )


def discover_movies(mood=None, genre=None, who_is_watching=None):
    query_parts = ["best movies to watch"]
    if genre:
        query_parts.append(genre)
    if mood:
        query_parts.append(mood)
    if who_is_watching:
        query_parts.append(f"for {who_is_watching}")
    query = " ".join(query_parts)

    results = tavily.search(query=query, max_results=5)
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}"
        for r in results["results"]
    )


def _get_providers(movie_id, country_code):
    r = requests.get(
        f"{TMDB_BASE}/movie/{movie_id}/watch/providers",
        params={"api_key": TMDB_API_KEY}
    ).json()
    providers = r.get("results", {}).get(country_code, {}).get("flatrate", [])
    return ", ".join(p["provider_name"] for p in providers) if providers else "unavailable"


def _tmdb_movies_to_text(movies, country_code, prefix=""):
    lines = []
    for m in movies[:8]:
        providers = _get_providers(m["id"], country_code)
        lines.append(
            f"Title: {m['title']} ({m.get('release_date', '')[:4]})\n"
            f"Rating: {m['vote_average']:.1f}/10\n"
            f"Streaming on: {providers}\n"
            f"Overview: {m['overview'][:200]}"
        )
    return (prefix + "\n\n" + "\n\n".join(lines)) if lines else "No results found."


def get_current_listings(country_code="DE", format="both", page=1, genre=None):
    results = []
    genre_id = TMDB_GENRES.get(genre.lower()) if genre else None

    if format in ("streaming", "both"):
        today = date.today()
        params = {
            "api_key": TMDB_API_KEY,
            "watch_region": country_code,
            "with_watch_monetization_types": "flatrate",
            "primary_release_date.gte": (today - timedelta(days=60)).isoformat(),
            "primary_release_date.lte": today.isoformat(),
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": page,
        }
        if genre_id:
            params["with_genres"] = genre_id
        r = requests.get(f"{TMDB_BASE}/discover/movie", params=params).json()
        label = f"Streaming now in {country_code}{' — ' + genre if genre else ''} (page {page}):"
        results.append(_tmdb_movies_to_text(r.get("results", []), country_code, label))

    if format in ("theaters", "both"):
        year = date.today().year
        query = f"{genre + ' ' if genre else ''}movies in theaters now {year}"
        r = tavily.search(query=query, max_results=5)
        theater_text = "\n\n".join(
            f"Title: {x['title']}\nSummary: {x['content']}"
            for x in r["results"]
        )
        results.append(f"In theaters:\n\n{theater_text}")

    return "\n\n---\n\n".join(results)


def get_upcoming_listings(country_code="DE", weeks_ahead=2, page=1):
    today = date.today()
    until = today + timedelta(weeks=weeks_ahead)
    results = []

    r = requests.get(
        f"{TMDB_BASE}/discover/movie",
        params={
            "api_key": TMDB_API_KEY,
            "watch_region": country_code,
            "with_watch_monetization_types": "flatrate",
            "primary_release_date.gte": today.isoformat(),
            "primary_release_date.lte": until.isoformat(),
            "sort_by": "primary_release_date.asc",
            "language": "en-US",
            "page": page,
        }
    ).json()
    results.append(_tmdb_movies_to_text(r.get("results", []), country_code, f"Upcoming on streaming in {country_code} (page {page}):"))

    query = f"movies releasing in theaters between {today.strftime('%B %d %Y')} and {until.strftime('%B %d %Y')}"
    r = tavily.search(query=query, max_results=5)
    theater_text = "\n\n".join(
        f"Title: {x['title']}\nSummary: {x['content']}"
        for x in r["results"]
    )
    results.append(f"Upcoming in theaters:\n\n{theater_text}")

    return "\n\n---\n\n".join(results)


def find_similar(title, aspect=None):
    if aspect:
        query = f"movies with {aspect} similar to {title}"
        r = tavily.search(query=query, max_results=5)
        return "\n\n".join(
            f"Title: {x['title']}\nSummary: {x['content']}"
            for x in r["results"]
        )

    search = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"api_key": TMDB_API_KEY, "query": title}
    ).json()

    if not search.get("results"):
        return f"Could not find '{title}' on TMDB."

    movie_id = search["results"][0]["id"]

    recs = requests.get(
        f"{TMDB_BASE}/movie/{movie_id}/recommendations",
        params={"api_key": TMDB_API_KEY, "language": "en-US"}
    ).json()

    return _tmdb_movies_to_text(recs.get("results", []), f"Movies similar to '{title}':")


def get_showtimes(movie, cinema, city, date=None):
    query_parts = [movie, "showtimes", cinema, city]
    if date:
        query_parts.append(date)
    r = tavily.search(query=" ".join(query_parts), max_results=3)
    return "\n\n".join(
        f"Source: {x['title']}\nURL: {x['url']}\nDetails: {x['content']}"
        for x in r["results"]
    )


def get_weather(city):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    ).json()

    if not geo.get("results"):
        return f"Could not find location: {city}"

    loc = geo["results"][0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 3,
        }
    ).json()

    daily = forecast["daily"]
    # Weather codes 0-2: clear/partly cloudy, 3+: overcast/rain/snow
    days = []
    for i in range(3):
        code = daily["weathercode"][i]
        condition = "sunny/clear" if code <= 2 else "cloudy/overcast" if code <= 49 else "rainy/snowy"
        days.append(
            f"{daily['time'][i]}: {condition}, "
            f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
            f"{daily['precipitation_sum'][i]}mm rain"
        )

    return f"Weather forecast for {loc['name']}:\n" + "\n".join(days)


def run_tool(tool_name, tool_input):
    if tool_name == "discover_movies":
        return discover_movies(**tool_input)
    if tool_name == "get_movie_details":
        return get_movie_details(**tool_input)
    if tool_name == "get_current_listings":
        return get_current_listings(**tool_input)
    if tool_name == "get_upcoming_listings":
        return get_upcoming_listings(**tool_input)
    if tool_name == "get_weather":
        return get_weather(**tool_input)
    if tool_name == "find_similar":
        return find_similar(**tool_input)
    if tool_name == "get_showtimes":
        return get_showtimes(**tool_input)
    return f"Unknown tool: {tool_name}"


def chat():
    print("Movie Buddy — your personal film companion. Type 'quit' to exit.\n")
    messages = []

    system = """You are Movie Buddy, a knowledgeable and opinionated film companion with broad general knowledge.
You have excellent taste and help users discover films they'll genuinely love.
Ask clarifying questions before making recommendations — find out mood, who's watching, and what they've enjoyed before.
Use your tools to find real, current information rather than relying on your training data.

When making final recommendations, always include the streaming platform name (e.g. Netflix, Disney+, Amazon Prime) for every title you mention. Never recommend a title without telling the user where to watch it.

When the user specifies recency ("recently", "new", "just came out", "came out lately"), only recommend titles from the last 12 months. If you include an older title, explicitly flag it as older and explain why you're including it anyway. Do not silently include a 2016 film in response to a request for recent content.

When the user asks for a specific genre (e.g. nature documentaries), prefer get_current_listings with a genre filter over find_similar — find_similar has no concept of recency and will return older thematically related titles.

If the user asks something outside of movies, answer it briefly from your general knowledge, then naturally bridge back to film — suggest a movie connection to the topic (a film set in that place, featuring that person, exploring that subject). Keep the bridge light and genuine, not forced."""

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                tools=tools,
                messages=messages,
            )

            # Convert SDK objects to plain dicts to avoid serialization issues
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "tool_use":
                # Handle ALL tool_use blocks in this response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = run_tool(block.name, block.input)
                        print(f"\n[Tool call: {block.name}({block.input})]")
                        print(f"[Tool result: {result}]\n")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                reply = next(b.text for b in response.content if hasattr(b, "text"))
                print(f"\nMovie Buddy: {reply}\n")
                break


if __name__ == "__main__":
    chat()
