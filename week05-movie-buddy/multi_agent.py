import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(__file__))

from agent import client, run_tool, tools as all_tools

try:
    from ddtrace.llmobs import LLMObs
    LLMOBS_ENABLED = True
except ImportError:
    LLMOBS_ENABLED = False

try:
    from datadog import statsd
    STATSD_ENABLED = True
except ImportError:
    STATSD_ENABLED = False

INJECTION_PATTERNS = [
    "ignore your previous instructions",
    "ignore all previous instructions",
    "forget everything above",
    "forget your instructions",
    "you are now",
    "act as dan",
    "pretend you are",
    "reveal your system prompt",
    "show me your prompt",
    "what are your instructions",
    "disregard your",
    "override your",
    "jailbreak",
    "do anything now",
]


def detect_prompt_injection(text):
    """Returns True if the text contains known prompt injection patterns."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)


def handle_injection(user_message):
    """Log the injection attempt to Datadog and return a safe response."""
    print(f"[SECURITY] Prompt injection detected: {user_message[:100]}")

    if STATSD_ENABLED:
        statsd.increment("movie_buddy.security.prompt_injection_blocked", tags=["service:movie-buddy"])

    return "I'm Movie Buddy — I can only help with movie recommendations, showtimes, and streaming suggestions. Is there a film I can help you find? 🎬"

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


def _run_specialist(system_prompt, tools, request, model="claude-sonnet-4-6"):
    """Spin up a specialist agent, run its tool loop, return its answer."""
    messages = [{"role": "user", "content": request}]

    while True:
        response = client.messages.create(
            model=model,
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
    start = time.time()
    print(f"[{name}] started")
    label = name.replace("ask_", "").replace("_", "-").title()

    def _run():
        if name == "ask_tracker":
            return _run_specialist(TRACKER_PROMPT, TRACKER_TOOLS, request, model="claude-sonnet-4-6")
        if name == "ask_explorer":
            return _run_specialist(EXPLORER_PROMPT, EXPLORER_TOOLS, request, model="claude-sonnet-4-6")
        if name == "ask_fact_checker":
            return _run_specialist(FACTCHECK_PROMPT, FACTCHECK_TOOLS, request, model="claude-haiku-4-5-20251001")
        if name == "ask_planner":
            return _run_specialist(PLANNER_PROMPT, PLANNER_TOOLS, request, model="claude-sonnet-4-6")
        return f"Unknown specialist: {name}"

    if LLMOBS_ENABLED:
        with LLMObs.agent(name=label) as span:
            LLMObs.annotate(span, input_data=[{"role": "user", "content": request}])
            result = _run()
            LLMObs.annotate(span, output_data=[{"role": "assistant", "content": result}])
    else:
        result = _run()

    elapsed = round(time.time() - start, 1)
    print(f"[{name}] finished in {elapsed}s")
    return result, elapsed


def process_message(messages):
    """Run one orchestrator turn. Appends to messages in place.
    Returns (reply, specialist_calls_log, total_specialist_elapsed_s)."""
    calls_log = []
    total_elapsed = None
    user_msg = next(
        (m["content"] for m in reversed(messages)
         if m["role"] == "user" and isinstance(m.get("content"), str)),
        ""
    )

    if detect_prompt_injection(user_msg):
        reply = handle_injection(user_msg)
        return reply, calls_log, total_elapsed

    ctx = LLMObs.workflow(name="Movie Buddy") if LLMOBS_ENABLED else nullcontext()
    with ctx as workflow_span:
        if LLMOBS_ENABLED:
            LLMObs.annotate(workflow_span, input_data=[{"role": "user", "content": user_msg}])

        reply = None
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
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                # Specialists in the same orchestrator response are independent by design —
                # the orchestrator only batches calls it can resolve simultaneously.
                # We parallelise them here to reduce latency.
                parallel_start = time.time()
                with ThreadPoolExecutor(max_workers=len(tool_use_blocks)) as executor:
                    futures = {
                        executor.submit(_call_specialist, block.name, block.input["request"]): block
                        for block in tool_use_blocks
                    }
                    results = {futures[f].id: f.result() for f in as_completed(futures)}
                total_elapsed = round(time.time() - parallel_start, 1)
                print(f"All specialists finished in {total_elapsed}s (parallel)")

                # Preserve original order when building tool_results so IDs match correctly
                tool_results = []
                for block in tool_use_blocks:
                    result, elapsed = results[block.id]
                    calls_log.append({
                        "specialist": block.name.replace("ask_", "").replace("_", "-").title(),
                        "request": block.input["request"],
                        "elapsed_s": elapsed,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                reply = next(b.text for b in response.content if hasattr(b, "text"))
                if LLMOBS_ENABLED and workflow_span is not None:
                    LLMObs.annotate(workflow_span, output_data=[{"role": "assistant", "content": reply}])
                break

    return reply, calls_log, total_elapsed
