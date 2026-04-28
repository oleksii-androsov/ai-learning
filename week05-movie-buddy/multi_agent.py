import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent import client, run_tool, tools as all_tools

# Pull each specialist's tools from the master list by name
def _tools_by_name(*names):
    return [t for t in all_tools if t["name"] in names]

TRACKER_TOOLS     = _tools_by_name("get_current_listings", "get_upcoming_listings")
EXPLORER_TOOLS    = _tools_by_name("discover_movies", "find_similar")
FACTCHECK_TOOLS   = _tools_by_name("get_movie_details")
PLANNER_TOOLS     = _tools_by_name("get_showtimes", "get_weather")

# Specialist system prompts
TRACKER_PROMPT = """You are Tracker, a specialist in finding what movies are currently available or coming soon.
You have access to current theater listings, streaming catalogs, and upcoming release schedules.
Return clear, factual information about availability — titles, release dates, platforms, country-accurate streaming data."""

EXPLORER_PROMPT = """You are Explorer, a specialist in discovering movies based on preferences.
You surface recommendations based on mood, genre, who is watching, and similarity to films the user has enjoyed.
Return a focused list of relevant titles with brief reasoning for each.
Only recommend titles appropriate for the youngest viewer when children are present."""

FACTCHECK_PROMPT = """You are Fact-Checker, a specialist in providing accurate factual information about specific films.
You look up precise details: cast, director, runtime, FSK age rating, TMDB score, plot overview.
Return structured, accurate facts. Always surface the FSK age rating clearly."""

PLANNER_PROMPT = """You are Planner, a specialist in logistics — showtimes and weather.
When both are relevant, correlate them: if weather is bad on the day the user wants to visit a cinema, flag it.
Only offer theater showtimes for films released in the last 8 weeks."""

# Orchestrator tools — each specialist is a tool the orchestrator can call
orchestrator_tools = [
    {
        "name": "ask_tracker",
        "description": "Ask the Tracker specialist what movies are currently showing in theaters or on streaming, or coming soon. Use when the user wants to know what's available now or in the near future.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The specific question for Tracker, with relevant context: country/city, format (theaters/streaming), and any time window."
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "ask_explorer",
        "description": "Ask the Explorer specialist to discover movies based on preferences, mood, genre, or similarity to films the user enjoyed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The discovery request, including user preferences, mood, who is watching, ages of children if relevant, and any reference films."
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "ask_fact_checker",
        "description": "Ask the Fact-Checker specialist for accurate details about a specific film: cast, director, runtime, FSK age rating, TMDB score, plot overview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The film title and what facts are needed."
                }
            },
            "required": ["request"]
        }
    },
    {
        "name": "ask_planner",
        "description": "Ask the Planner specialist about showtimes at a specific cinema, weather forecast, or both. Planner correlates weather with showtime planning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The logistics request — cinema name, city, film title, and any specific date."
                }
            },
            "required": ["request"]
        }
    }
]

ORCHESTRATOR_PROMPT = """You are Movie Buddy, a knowledgeable and opinionated film companion.
You have a team of four specialists you can consult:

- Tracker: finds what's currently in theaters or on streaming, and what's coming soon
- Explorer: discovers movies based on preferences, mood, genre, or similarity to films the user enjoyed
- Fact-Checker: provides accurate details about specific films (cast, rating, FSK age certification, runtime)
- Planner: handles logistics — showtimes at specific cinemas, weather forecasts, and correlating the two

Your job: understand what the user needs, call the right specialist(s), and synthesize their findings into a clear, helpful response. You may call multiple specialists when a request spans domains.

Ask clarifying questions before consulting specialists when you're missing key information — location, who's watching, ages of children, mood.

Never state or imply a film's release status from memory — always verify via Tracker or Fact-Checker. A film you think is "coming soon" may already be in cinemas.

Always include streaming platform names in recommendations. Mirror the user's communication style."""


def _run_specialist(system_prompt, tools, request):
    """Spin up a specialist agent, run its tool loop, return its answer."""
    messages = [{"role": "user", "content": request}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

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
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            return next(b.text for b in response.content if hasattr(b, "text"))


def _call_specialist(name, request):
    if name == "ask_tracker":
        return _run_specialist(TRACKER_PROMPT, TRACKER_TOOLS, request)
    if name == "ask_explorer":
        return _run_specialist(EXPLORER_PROMPT, EXPLORER_TOOLS, request)
    if name == "ask_fact_checker":
        return _run_specialist(FACTCHECK_PROMPT, FACTCHECK_TOOLS, request)
    if name == "ask_planner":
        return _run_specialist(PLANNER_PROMPT, PLANNER_TOOLS, request)
    return f"Unknown specialist: {name}"


def process_message(messages):
    """Run one orchestrator turn. Appends to messages in place.
    Returns (reply, specialist_calls_log)."""
    calls_log = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=ORCHESTRATOR_PROMPT,
            tools=orchestrator_tools,
            messages=messages,
        )

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
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _call_specialist(block.name, block.input["request"])
                    calls_log.append({
                        "specialist": block.name.replace("ask_", "").replace("_", "-").title(),
                        "request": block.input["request"],
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            reply = next(b.text for b in response.content if hasattr(b, "text"))
            return reply, calls_log
