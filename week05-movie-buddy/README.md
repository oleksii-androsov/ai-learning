# Week 5 — Movie Buddy: AI Agent with Tool Use

## What we're building

A conversational movie recommendation agent that uses real-time tools to give personalised, current suggestions — not just what Claude knows from training data. The agent asks the right questions, searches the web, fetches movie details, and reasons across multiple sources before making a recommendation.

This is fundamentally different from asking ChatGPT for movie suggestions:
- **No hallucinated titles** — it searches for real, current information
- **No stale data** — it knows what's in theaters this week
- **Personalised** — it learns your taste, family situation, and mood before suggesting
- **Self-correcting** — it can verify its own answers when challenged

## Architecture

```
User (terminal)
    ↓ conversation
Agent loop (agent.py)
    ↓ decides which tools to call
Tools
    ├── discover_movies → Tavily web search
    ├── get_movie_details → TMDB API (Day 2)
    ├── get_current_listings → Tavily web search (Day 3)
    ├── get_upcoming_listings → Tavily web search (Day 3)
    ├── get_weather → weather API (Day 3)
    └── find_similar → TMDB + Tavily (Day 4)
    ↓ results fed back to Claude
Claude reasons across all results → final recommendation
```

## How tool use works

The agent runs an inner loop: Claude responds → if it wants a tool, we execute it and feed the result back → Claude continues reasoning → repeat until Claude has enough to answer.

Claude decides autonomously:
- **Which tool to call** — based on the tool descriptions in the code
- **When to call multiple tools** — it may call `discover_movies` three times with different genres
- **When to stop** — once it has enough information to give a good answer
- **When not to use tools** — simple questions get direct answers without any tool calls

## Getting started

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   ANTHROPIC_API_KEY=...
   TAVILY_API_KEY=...     # from tavily.com
   TMDB_API_KEY=...       # from themoviedb.org
   ```
3. Install dependencies:
   ```bash
   pip install -r week05-movie-buddy/requirements.txt
   ```
4. Run:
   ```bash
   python week05-movie-buddy/agent.py
   ```

## Day-by-day progress

### Day 4 — find_similar, get_showtimes, streaming improvements
- `find_similar(title, aspect)` — without `aspect`: TMDB `/movie/{id}/recommendations` for thematically similar films; with `aspect` (e.g. "Hans Zimmer soundtrack", "Andreas Deja animation"): Tavily web search for that specific creative angle. Same tool, two data sources.
- `get_showtimes(movie, cinema, city, date)` — Tavily search for specific cinema schedules including optional date. Tested with Astor Film Lounge MyZeil in Frankfurt.
- Streaming now uses TMDB `discover/movie` with `watch_region` — country-accurate results across Netflix, Prime, Disney+, Apple TV+ etc. Provider info fetched per title.
- Added `genre` filter to `get_current_listings` — Claude can now search "nature documentaries streaming now in Germany" using TMDB genre IDs instead of falling back to `find_similar`
- Added `page` parameter to listing tools — "give me more options" correctly fetches the next page rather than repeating page 1
- System prompt improvements: always include platform in recommendations; flag titles older than 12 months when user requests recent content

### Day 3 — get_current_listings, get_upcoming_listings, get_weather
- `get_current_listings(location, format)` — Tavily search for what's in theaters or on streaming right now
- `get_upcoming_listings(location, weeks_ahead)` — Tavily search for upcoming releases, good for planning ahead
- `get_weather(city)` — Open-Meteo API (no key needed): geocodes the city, fetches 3-day forecast, maps weather codes to human-readable conditions
- Agent now combines weather + listings in a single turn: "cloudy but dry → still worth going out, here's what's showing"
- Gap identified: agent correctly declines to check specific cinema showtimes — needs a dedicated `get_showtimes` tool (planned for Day 4)

### Day 2 — get_movie_details tool via TMDB
- Added `get_movie_details(title)` — searches TMDB by title, fetches full details with cast in a single call using `append_to_response=credits`
- Returns rating, runtime, genres, director, top 5 cast, and plot overview
- TMDB requires the short API Key (not the Read Access Token) for query parameter auth
- Agent now chains both tools naturally: discovers candidates with Tavily, then fetches structured facts from TMDB for specific titles

### Day 1 — Core agent loop + discover_movies tool
- Built the tool use conversation loop: Claude calls tools, we execute them, feed results back, repeat until `end_turn`
- First tool: `discover_movies(mood, genre, who_is_watching)` — Tavily web search returning current results
- Agent asks clarifying questions before searching — mood, who's watching, recent films enjoyed
- Handles multiple tool calls per turn (Claude called `discover_movies` 3× in one response, once per genre)
- Verified it self-corrects: when challenged on Project Hail Mary's release date, it searched and corrected itself
- Fetched age-appropriate guidance without being explicitly asked — inferred from context

## Key architectural decisions

- **Tool descriptions drive behaviour** — Claude decides when and how to call tools based entirely on the description text. Writing good descriptions is as important as writing good code.
- **All tool_use blocks handled per response** — Claude can call multiple tools in a single response; we loop over all of them and return all results before Claude continues.
- **Plain dicts over SDK objects** — assistant message content is converted to plain dicts before appending to the message history, avoiding SDK serialization edge cases.
- **Tavily over raw web search** — returns clean structured results designed for LLM consumption, not raw HTML.

## What's coming

- **Day 5** — Streamlit UI, deploy to EC2
