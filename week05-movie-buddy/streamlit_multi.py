import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from multi_agent import process_message

st.set_page_config(page_title="Movie Buddy", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .movie-buddy-header { text-align: center; padding: 1.5rem 0 0.75rem 0; }
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

if "messages"        not in st.session_state: st.session_state.messages        = []
if "display_history" not in st.session_state: st.session_state.display_history = []
if "pending_prompt"  not in st.session_state: st.session_state.pending_prompt  = None


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


# Welcome screen — only when no conversation and nothing pending
if not st.session_state.display_history and not st.session_state.pending_prompt:
    st.markdown("#### What can I help you find?")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

# Render conversation history
for i, entry in enumerate(st.session_state.display_history):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
    if entry.get("calls"):
        render_expander(entry["calls"], entry.get("total_elapsed"), key=f"expander_{i}")

# Chat input — example button clicks feed through pending_prompt
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
            reply, calls, total_elapsed = process_message(st.session_state.messages)
        st.markdown(reply)
        if calls:
            render_expander(calls, total_elapsed, key=f"expander_{len(st.session_state.display_history)}")

    st.session_state.display_history.append({
        "role":          "assistant",
        "content":       reply,
        "calls":         calls,
        "total_elapsed": total_elapsed,
    })
