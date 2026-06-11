import hashlib
from typing import Optional
import streamlit as st


def _email_to_user_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def resolve_user(session_state) -> Optional[str]:
    return session_state.get("user_id")


def show_auth_sidebar(session_state):
    """Render auth UI in sidebar. Sets session_state.user_id when resolved."""
    if session_state.get("user_id"):
        return

    with st.sidebar:
        st.markdown("### Sign in to Movie Buddy")
        st.markdown("Enter your email to load your preferences.")

        email = st.text_input("Email address", key="auth_email")
        if st.button("Continue", key="auth_submit"):
            if "@" not in email:
                st.error("Please enter a valid email.")
                return
            session_state["user_id"] = _email_to_user_id(email)
            st.rerun()
