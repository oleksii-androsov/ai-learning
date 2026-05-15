import hashlib
import uuid
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components
from memory import get_user_id_for_device, register_device

STORAGE_KEY = "mb_device_token"


def _get_stored_token() -> Optional[str]:
    """Read device token from localStorage via JS component."""
    token = st.session_state.get("_ls_device_token")
    if token:
        return token

    result = components.html(
        f"""
        <script>
        const token = localStorage.getItem("{STORAGE_KEY}") || "";
        const input = window.parent.document.querySelector('input[data-testid="stTextInput-ls_token"]');
        if (input && token) {{
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(input, token);
            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
        </script>
        """,
        height=0,
    )
    return None


def _set_stored_token(token: str):
    """Write device token to localStorage."""
    components.html(
        f"""
        <script>
        localStorage.setItem("{STORAGE_KEY}", "{token}");
        </script>
        """,
        height=0,
    )


def _email_to_user_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def resolve_user(session_state) -> Optional[str]:
    """Return user_id if known this session, else None."""
    return session_state.get("user_id")


def show_auth_sidebar(session_state):
    """Render auth UI in sidebar. Sets session_state.user_id when resolved."""
    if session_state.get("user_id"):
        return

    with st.sidebar:
        st.markdown("### Sign in to Movie Buddy")
        st.markdown("Enter your email to save preferences across sessions.")

        # Hidden text input used to receive localStorage token via JS
        ls_token = st.text_input(
            "ls_token",
            key="ls_token",
            label_visibility="collapsed",
        )
        if ls_token and ls_token != session_state.get("_ls_device_token"):
            session_state["_ls_device_token"] = ls_token
            user_id = get_user_id_for_device(ls_token)
            if user_id:
                session_state["user_id"] = user_id
                st.rerun()

        _get_stored_token()

        email = st.text_input("Email address", key="auth_email")
        if st.button("Continue", key="auth_submit"):
            if "@" not in email:
                st.error("Please enter a valid email.")
                return
            user_id = _email_to_user_id(email)
            device_token = str(uuid.uuid4())
            register_device(device_token, user_id)
            session_state["user_id"] = user_id
            _set_stored_token(device_token)
            st.rerun()
