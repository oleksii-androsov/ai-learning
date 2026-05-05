# Movie Buddy Week 6 — Multi-Agent Orchestration + Datadog Observability

Continuation of Movie Buddy from Week 5. This week redesigns the single agent into an orchestrator + specialist architecture, adds full Datadog LLM Observability instrumentation, prompt injection security, and UI polish.

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

The multi-agent code lives here in `week06-movie-buddy/`. It imports tool functions from `week05-movie-buddy/agent.py` — no duplication of tool implementations.

## Week 6 progress

### Day 1 — Multi-agent orchestration
- Redesigned from single agent to orchestrator + four specialists: Tracker, Explorer, Fact-Checker, Planner
- Orchestrator has no tools — pure reasoning. Decides which specialist to call, what to ask, synthesises the final response
- Each specialist runs its own tool loop with a focused subset of tools and a tight system prompt
- Specialists are the orchestrator's "tools" — same tool use loop pattern, one level up
- `streamlit_multi.py` — new UI pointing at the orchestrator, shows which specialists were consulted per turn

### Day 2 — Parallel execution, model routing, Datadog LLM Observability, HTTPS
- Parallel specialist execution using `ThreadPoolExecutor` — independent specialists fire simultaneously, total latency = max(individual), not sum
- Per-specialist timing visible in Streamlit expander and console
- Datadog LLM Observability instrumentation — `LLMObs.workflow` spans for orchestrator, `LLMObs.agent` spans for each specialist, auto-instrumented Anthropic API calls as child spans
- Full trace waterfall visible in Datadog: workflow → specialist agents → LLM calls, with token counts, cost, and latency per span
- Model routing: Haiku for Fact-Checker (structured lookups), Sonnet for Tracker/Explorer/Planner/Orchestrator (reasoning-heavy)
- nginx reverse proxy + Let's Encrypt SSL — app live at **https://movie-buddy.app**

### Day 3 — Prompt injection detection
- Added security layer: every user message passes through a Haiku semantic classifier before reaching the orchestrator
- Classifier prompt: 5-line YES/NO prompt — reads intent, not keywords. Catches novel phrasings the old keyword list would miss
- Fail-open on error — won't block legitimate users if Haiku errors
- Blocked attempts: logged to console `[SECURITY]`, statsd metric incremented (`movie_buddy.security.prompt_injection_blocked`), LLMObs span created with `metadata: {security: {injection_blocked: True}}`
- Filterable in LLM Observability traces via `@metadata.security.injection_blocked:true`

### Day 4 — Datadog Dashboard + Monitors
- Live dashboard: Requests per minute, Avg response latency by specialist, LLM cost today, Prompt injection attempts blocked
- 3 monitors: High response latency (p95 > 45s), LLM cost spike (> $1/hour), Injection attack cluster (> 2 in 5 min)
- Key: use `.as_count()` on statsd counter metrics in monitors — without it Datadog shows fractional rate values

### Day 5 — UI polish + movie posters
- Gradient header, colour-coded specialist badges in expander
- Welcome screen with 4 clickable example prompts
- Movie poster display: Orchestrator appends `POSTERS: Title` to responses for 1-5 featured films; parsed in `process_message`, fetched from TMDB at w780 resolution, rendered inline
- Horizontal rule dividers stripped from responses

## Key architectural decisions

- **Orchestrator has no tools** — pure reasoning layer. Adding tools to the orchestrator would blur the separation of concerns and make it harder to debug which specialist is responsible for what.
- **Specialists are stateless** — each specialist call is independent, enabling parallel execution.
- **Model routing by task type** — Haiku for structured fact lookups (Fact-Checker), Sonnet for reasoning-heavy tasks.
- **LLM-as-judge for security** — Haiku classifier upstream of the main system reads intent, not keywords. Catches novel injection phrasings a static list would miss.
- **Poster hints via prompt, not tool** — Orchestrator appends a `POSTERS:` line rather than calling a fetch tool, avoiding an extra API round trip.
