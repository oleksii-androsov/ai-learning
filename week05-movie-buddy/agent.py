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
        "description": "Search for movies currently showing in theaters or recently released on streaming services. Use this when the user wants to know what's available to watch right now.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or country to check listings for, e.g. 'Frankfurt', 'Germany', 'UK'"
                },
                "format": {
                    "type": "string",
                    "description": "Where to watch: 'theaters', 'streaming', or 'both'",
                    "enum": ["theaters", "streaming", "both"]
                }
            },
            "required": []
        }
    },
    {
        "name": "get_upcoming_listings",
        "description": "Search for movies coming soon to theaters or streaming. Use this when the user wants to plan ahead or know what's releasing in the coming weeks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or country to check listings for, e.g. 'Frankfurt', 'Germany'"
                },
                "weeks_ahead": {
                    "type": "integer",
                    "description": "How many weeks ahead to look, e.g. 2 for the next two weeks"
                }
            },
            "required": []
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


def get_current_listings(location=None, format="both"):
    query_parts = ["movies"]
    if format == "theaters":
        query_parts.append("in theaters now")
    elif format == "streaming":
        query_parts.append("new on streaming services")
    else:
        query_parts.append("in theaters and new on streaming")
    if location:
        query_parts.append(location)
    year = date.today().year
    query_parts.append(f"{year} {year + 1}")

    results = tavily.search(query=" ".join(query_parts), max_results=5)
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}"
        for r in results["results"]
    )


def get_upcoming_listings(location=None, weeks_ahead=2):
    today = date.today()
    until = today + timedelta(weeks=weeks_ahead)
    query_parts = [
        "movies releasing in theaters or streaming",
        f"between {today.strftime('%B %d %Y')} and {until.strftime('%B %d %Y')}"
    ]
    if location:
        query_parts.append(location)

    results = tavily.search(query=" ".join(query_parts), max_results=5)
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}"
        for r in results["results"]
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
    return f"Unknown tool: {tool_name}"


def chat():
    print("Movie Buddy — your personal film companion. Type 'quit' to exit.\n")
    messages = []

    system = """You are Movie Buddy, a knowledgeable and opinionated film companion with broad general knowledge.
You have excellent taste and help users discover films they'll genuinely love.
Ask clarifying questions before making recommendations — find out mood, who's watching, and what they've enjoyed before.
Use your tools to find real, current information rather than relying on your training data.

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
