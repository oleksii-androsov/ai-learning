import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from multi_agent import process_message

st.set_page_config(page_title="Movie Buddy", page_icon="🎬", layout="centered")
st.title("🎬 Movie Buddy")
st.caption("Tracker · Explorer · Fact-Checker · Planner")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []


def render_expander(calls, total_elapsed, key):
    header = f"🤖 {len(calls)} specialist(s) consulted"
    if total_elapsed is not None:
        header += f" · {total_elapsed}s total"
    with st.expander(header, key=key):
        for c in calls:
            st.caption(f"**{c['specialist']}** ({c.get('elapsed_s', '?')}s) — {c['request']}")


for i, entry in enumerate(st.session_state.display_history):
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
    if entry.get("calls"):
        render_expander(entry["calls"], entry.get("total_elapsed"), key=f"expander_{i}")

if prompt := st.chat_input("Ask about movies..."):
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
        "role": "assistant",
        "content": reply,
        "calls": calls,
        "total_elapsed": total_elapsed,
    })
