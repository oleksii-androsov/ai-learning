# Week 1 — Python Basics & First Claude API Call

## What we built

An interactive command-line chatbot that sends questions to Claude via the Anthropic API and prints responses. Runs as a loop until you type `quit`.

## Getting started

1. Clone the repo and navigate into it
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install dependencies: `pip install anthropic python-dotenv`
5. Copy the env template: `cp .env.example .env`
6. Add your Anthropic API key to `.env`
7. Run: `python3 week01-python-basics/hello_claude.py`

## Key architectural decisions

- **Environment variables for secrets** — API key is loaded from a `.env` file via `python-dotenv`, never hardcoded. `.env` is in `.gitignore`.
- **`while True` loop with `break`** — keeps the session alive across multiple questions without restarting the script.
- **Stateless API calls** — each question is a fresh conversation. The API has no memory between calls; context management is the application's responsibility.

## What I learned

- How to set up a Python virtual environment and why it matters
- How to install and use a third-party library (`anthropic`, `python-dotenv`)
- How to read environment variables in Python with `os.environ`
- That LLMs are stateless — every API call starts from zero unless you pass conversation history yourself

## What I'd do differently

- Add conversation history so follow-up questions have context (Day 2)
- Add error handling for missing API key or network failures (Day 3)
