import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_KEY = os.environ["RAG_API_KEY"]

st.title("RAG Document Assistant")
st.caption("Ask questions about the loaded document.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])
    if uploaded_file and st.button("Index document"):
        with st.spinner("Indexing..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    headers={"X-API-Key": API_KEY},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                st.success(data["message"])
                st.caption(f"{data['chunks']} chunks indexed.")
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "history": history},
                    headers={"X-API-Key": API_KEY},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                if not data["sources_found"]:
                    answer = "⚠️ No relevant content found in the document — answering from general knowledge.\n\n" + answer
            except requests.exceptions.Timeout:
                answer = "Request timed out. The API may be busy — try again."
            except requests.exceptions.RequestException as e:
                answer = f"Could not reach the API: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
