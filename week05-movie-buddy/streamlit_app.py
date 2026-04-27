import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from agent import process_message

st.set_page_config(page_title="Movie Buddy", page_icon="🎬", layout="centered")
st.title("🎬 Movie Buddy")
st.caption("Your personal film companion — recommendations, showtimes, streaming, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user" and isinstance(content, str):
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant" and isinstance(content, list):
        text = next((b["text"] for b in content if b.get("type") == "text"), None)
        if text:
            with st.chat_message("assistant"):
                st.markdown(text)

if prompt := st.chat_input("Ask about movies, showtimes, streaming..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, tool_calls = process_message(st.session_state.messages)

        st.markdown(reply)

        if tool_calls:
            with st.expander(f"🔧 {len(tool_calls)} tool call(s)"):
                for tc in tool_calls:
                    st.caption(f"**{tc['tool']}** — {tc['input']}")
