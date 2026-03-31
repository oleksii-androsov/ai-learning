# Week 1 — Python Basics & First Claude API Call

## What we built

An interactive command-line chatbot that:
- Sends questions to Claude via the Anthropic API
- Maintains full conversation history so follow-up questions have context
- Loads a text document and uses it as context for answers (primitive RAG)
- Handles errors gracefully — missing files, bad API keys, connection failures
- Is structured into functions for readability and maintainability

## Getting started

1. Clone the repo and navigate into it
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install dependencies: `pip install anthropic python-dotenv`
5. Copy the env template: `cp .env.example .env`
6. Add your Anthropic API key to `.env`
7. Run: `python3 week01-python-basics/hello_claude.py`

## Key architectural decisions

- **Environment variables for secrets** — API key loaded from `.env` via `python-dotenv`, never hardcoded. `.env` is in `.gitignore`. `.env.example` is committed as a template for others.
- **Conversation history as a list of dicts** — the Anthropic API is stateless. To simulate memory, every user message and assistant reply is appended to a list and sent with each API call.
- **System prompt for document context** — the loaded document is passed via the `system` parameter, which persists across the whole session without cluttering the conversation history.
- **`None` as error signal** — error handlers print messages and return `None` rather than returning error strings. This avoids ambiguity since a real Claude reply could start with "Error:".
- **Functions for separation of concerns** — `load_document()`, `chat()`, and `main()` each own one responsibility, making the code easier to debug and extend in Week 2.

## What I learned

- How to set up a Python virtual environment and why it matters
- How to install and use third-party libraries (`anthropic`, `python-dotenv`)
- How to read environment variables securely with `os.environ`
- That LLMs are stateless — every API call starts from zero unless you pass conversation history yourself
- Python lists and dictionaries — the core data structures behind the API's message format
- How to read a file in Python and pass its contents as prompt context
- The difference between `system` prompt and conversation messages
- Basic error handling with `try/except` and early `return`

## What I'd do differently

- Make the document path configurable (accept it as a command-line argument)
- Add persistent conversation history across sessions (save/load from a file)
- Week 2 will replace the single-file approach with proper chunking and vector search for larger documents
