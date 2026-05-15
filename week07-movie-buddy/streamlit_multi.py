import sys
import os
import re
import logging
import threading
import requests

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv
from multi_agent import process_message
from auth import resolve_user, show_auth_sidebar
from memory import (
    get_profile, save_profile, empty_profile,
    get_summary, save_summary,
    format_profile_for_prompt, format_summary_for_prompt,
)
from profile_extractor import extract_profile_updates, update_summary, should_extract

load_dotenv()

st.set_page_config(page_title="Movie Buddy", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .movie-buddy-header {
        text-align: center; padding: 1.75rem 2rem 1.25rem 2rem;
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border-radius: 14px; margin-bottom: 1rem;
    }
    .movie-buddy-title  { font-size: 2.8rem; font-weight: 700; margin: 0; }
    .movie-buddy-sub    { color: #888; font-size: 0.9rem; margin-top: 0.25rem; letter-spacing: 0.05em; }
    .specialist-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; color: white; margin-right: 6px;
    }
    div[data-testid="stButton"] > button {
        text-align: left; height: auto; white-space: normal;
        padding: 0.65rem 1rem; border-radius: 10px; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="movie-buddy-header">
    <div class="movie-buddy-title">🎬 Movie Buddy</div>
    <div class="movie-buddy-sub">Tracker &nbsp;·&nbsp; Explorer &nbsp;·&nbsp; Fact-Checker &nbsp;·&nbsp; Planner</div>
</div>
""", unsafe_allow_html=True)

SPECIALIST_COLORS = {
    "Tracker":      "#1f77b4",
    "Explorer":     "#2ca02c",
    "Fact-Checker": "#ff7f0e",
    "Planner":      "#9467bd",
}

EXAMPLE_PROMPTS = [
    "What's a good film for a family with kids aged 8 and 12?",
    "What's currently showing in Frankfurt cinemas?",
    "I loved Interstellar — what should I watch next?",
    "What's new on Netflix Germany this week?",
]

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH  = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG     = "https://image.tmdb.org/t/p/w780"


def fetch_posters(titles):
    if not TMDB_API_KEY or not titles:
        return {}

    def _fetch(title):
        query = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
        try:
            r = requests.get(
                TMDB_SEARCH,
                params={"api_key": TMDB_API_KEY, "query": query, "language": "en-US"},
                timeout=5,
            )
            results = r.json().get("results", []) if r.ok else []
            if results:
                top = results[0]
                result_title = top.get("title", "").lower()
                query_lower  = query.lower()
                if (result_title == query_lower or query_lower in result_title):
                    if top.get("poster_path"):
                        return title, TMDB_IMG + top["poster_path"]
        except Exception:
            pass
        return title, None

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_fetch, t) for t in titles]
        return {t: url for f in as_completed(futures)
                for t, url in [f.result()] if url}


def render_reply(text, posters):
    text = re.sub(r'\n\s*---\s*\n', '\n\n', text)
    if not posters:
        st.markdown(text)
        return
    paragraphs = text.split("\n\n")
    used = set()
    for para in paragraphs:
        st.markdown(para)
        for title, url in posters.items():
            if title not in used and title in para:
                st.image(url, use_container_width=True)
                st.caption(title)
                used.add(title)


def _auto_save_memory(user_id, messages, profile, prior_summary):
    """Run in background thread — extract profile updates and save if anything changed."""
    try:
        updated_profile, changed = extract_profile_updates(messages, profile)
        if changed:
            save_profile(user_id, updated_profile)
            new_summary = update_summary(user_id, messages, prior_summary)
            save_summary(user_id, new_summary)
    except Exception as e:
        logger.warning(f"Auto-save memory failed: {e}")


def render_expander(calls, total_elapsed, key):
    header = f"🤖 {len(calls)} specialist(s) consulted"
    if total_elapsed is not None:
        header += f" · {total_elapsed}s"
    with st.expander(header, key=key):
        for c in calls:
            color = SPECIALIST_COLORS.get(c["specialist"], "#666")
            badge = f'<span class="specialist-badge" style="background:{color}">{c["specialist"]}</span>'
            st.markdown(
                f'{badge}<span style="color:#888;font-size:0.82rem">{c.get("elapsed_s","?")}s</span>'
                f' — {c["request"]}',
                unsafe_allow_html=True,
            )


# ---------- Session state init ----------

for key, default in [
    ("messages", []),
    ("display_history", []),
    ("pending_prompt", None),
    ("profile_loaded", False),
    ("profile", {}),
    ("prior_summary", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Auth ----------

show_auth_sidebar(st.session_state)
user_id = resolve_user(st.session_state)

# ---------- Load profile once per session ----------

if user_id and not st.session_state.profile_loaded:
    profile = get_profile(user_id)
    if not profile:
        profile = empty_profile(user_id)
        save_profile(user_id, profile)
    st.session_state.profile = profile
    st.session_state.prior_summary = get_summary(user_id)
    st.session_state.profile_loaded = True

# ---------- Build user context for Orchestrator ----------

user_context = ""
if user_id and st.session_state.profile:
    parts = []
    profile_text = format_profile_for_prompt(st.session_state.profile)
    summary_text = format_summary_for_prompt(st.session_state.prior_summary)
    if profile_text:
        parts.append(profile_text)
    if summary_text:
        parts.append(summary_text)
    user_context = "\n\n".join(parts)

# ---------- Sidebar ----------

with st.sidebar:
    if user_id:
        st.markdown("**Signed in** ✓")

# ---------- Main chat UI ----------

if not st.session_state.display_history and not st.session_state.pending_prompt:
    st.markdown("#### What can I help you find?")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

for i, entry in enumerate(st.session_state.display_history):
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant":
            render_reply(entry["content"], entry.get("posters", {}))
        else:
            st.markdown(entry["content"])
    if entry.get("calls"):
        render_expander(entry["calls"], entry.get("total_elapsed"), key=f"hist_expander_{i}")

prompt = st.chat_input("Ask about movies...")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.display_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the team..."):
            reply, calls, total_elapsed, poster_titles = process_message(
                st.session_state.messages,
                user_context=user_context,
            )
        posters = fetch_posters(poster_titles) if poster_titles else {}
        render_reply(reply, posters)
        if calls:
            render_expander(calls, total_elapsed, key=f"live_expander_{len(st.session_state.display_history)}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.display_history.append({
        "role":          "assistant",
        "content":       reply,
        "calls":         calls,
        "total_elapsed": total_elapsed,
        "posters":       posters,
    })

    # Auto-save memory in background if message looks profile-worthy
    if user_id and should_extract(prompt):
        threading.Thread(
            target=_auto_save_memory,
            args=(user_id, st.session_state.messages, st.session_state.profile, st.session_state.prior_summary),
            daemon=True,
        ).start()
