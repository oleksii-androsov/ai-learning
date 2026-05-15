# Movie Buddy Week 7 — Weather Fix + Memory, State, Guardrails

Continuation of Movie Buddy from Week 6. This week starts with fixing the weather tool, then adds DynamoDB persistent memory, preference inference, and output-side guardrails.

**Live demo:** https://movie-buddy.app

## Architecture

```
User (Streamlit)
    ↓
Orchestrator (claude-sonnet-4-6)
    ↓ delegates via tool calls
    ├── Tracker (claude-sonnet-4-6) — current listings, upcoming releases
    ├── Explorer (claude-sonnet-4-6) — discovery, recommendations, similar films
    ├── Fact-Checker (claude-haiku-4-5-20251001) — cast, rating, runtime, FSK
    └── Planner (claude-sonnet-4-6) — showtimes, weather, logistics
```

Specialists run in parallel via `ThreadPoolExecutor`. Total latency = slowest specialist, not the sum.

The multi-agent code lives here in `week07-movie-buddy/`. It imports tool functions from `week05-movie-buddy/agent.py` — no duplication of tool implementations.

## Week 7 progress

### Day 1 — Weather tool fixes

Three bugs fixed in the weather → recommendation pipeline:

**Bug 1 — Date hallucination:** The Orchestrator had no awareness of the current date, so when users asked about "this weekend" it would hallucinate dates months in the future. Open-Meteo only provides a 7-day forecast, so requests for hallucinated dates returned nothing. Fix: inject today's date and day name into the Orchestrator system prompt at runtime.

**Bug 2 — Temperature reporting:** The weather tool returns min and max temperatures per day. The Orchestrator was summarising with the overnight low, which contradicts how users read weather (their phone shows the daytime high). Fix: explicit prompt instruction to always reference the daytime high.

**Bug 3 — Wrong cinema/streaming logic:** The Orchestrator was rationalising rainy weather as a reason to go to the cinema ("you're not missing any sunshine!"). Fix: strict prompt rule — rain or snow means recommend streaming; dry pleasant weather means suggest cinema since it can be combined with a walk outside.

**Key lesson:** LLMs will find creative ways around vague instructions. "Rain means streaming" is not the same as "rain means streaming, never rationalise rain as a cinema reason." The rule needs to leave no room for interpretation.

### Day 2 — DynamoDB persistent memory

Three new modules: `memory.py` (DynamoDB read/write), `profile_extractor.py` (Haiku-based fact extraction), `auth.py` (email-based identity). Three DynamoDB tables: `movie_buddy_profiles`, `movie_buddy_summaries`, `movie_buddy_devices`.

**How it works:** After each assistant reply, a heuristic checks if the user's message is likely to contain profile-worthy information (trigger words: "liked", "watched", "kids", "netflix", etc., minimum 5 words). If yes, Haiku reads the full conversation and extracts structured facts — movies watched with opinions, genre preferences, children's ages, streaming platforms, weather preference. The extracted facts are merged into the existing profile and written to DynamoDB. The Orchestrator gets the profile and a rolling conversation summary injected into its system prompt at session start.

**Bugs fixed along the way:**
- DynamoDB silently rejects Python `None` values with a `TypeError` not caught by `except (BotoCoreError, ClientError)` — fixed by stripping `None` values before writing and catching all exceptions
- Profile overwrite bug: loading a non-existent profile created and saved an empty one, wiping any existing data — fixed by never saving on load
- Haiku date hallucination: asked to record `as_of` date for children's ages, it wrote `2025-01-10` — fixed by stamping the date in Python, not asking the model
- Shallow copy bug: `dict(existing)` shares nested list objects, so merged profile always equalled the original — `changed` was always `False` and nothing ever wrote to DynamoDB — fixed with `copy.deepcopy()`
- Cookie persistence: three approaches failed (streamlit-cookies-controller, st.context.cookies + JS iframe, URL query params). Root cause: Streamlit component iframes can't write to the parent page's cookie jar. Temporary fix: ask for email each session, persist profile by email hash. Auth0 integration planned next.

**Memory sidebar:** profile data visible in real time — genres, platforms, kids' ages (computed from stated age + date stated, not birth year), weather preference, movies watched count. Changes detected this session shown as a live feed.

## Key architectural decisions (inherited from Week 6)

- **Orchestrator has no tools** — pure reasoning layer. Adding tools to the orchestrator would blur the separation of concerns and make it harder to debug which specialist is responsible for what.
- **Specialists are stateless** — each specialist call is independent, enabling parallel execution.
- **Model routing by task type** — Haiku for structured fact lookups (Fact-Checker), Sonnet for reasoning-heavy tasks.
- **LLM-as-judge for security** — Haiku classifier upstream of the main system reads intent, not keywords. Catches novel injection phrasings a static list would miss.
- **Poster hints via prompt, not tool** — Orchestrator appends a `POSTERS:` line rather than calling a fetch tool, avoiding an extra API round trip.
- **Runtime date injection** — Today's date and day name injected into Orchestrator prompt on every turn so relative time references ("this weekend", "tomorrow") resolve correctly.
