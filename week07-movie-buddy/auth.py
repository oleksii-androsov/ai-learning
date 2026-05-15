import hashlib
import uuid
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components
from memory import get_user_id_for_device, register_device

COOKIE_NAME = "mb_device_token"
COOKIE_MAX_AGE_DAYS = 365


def _get_cookie() -> Optional[str]:
    """Read device token from HTTP request cookies — native Streamlit, no JS needed."""
    return st.context.cookies.get(COOKIE_NAME)


def _set_cookie(value: str):
    """Write cookie via JavaScript. Runs once; Streamlit reruns pick up the new cookie."""
    components.html(
        f"""
        <script>
        var d = new Date();
        d.setTime(d.getTime() + ({COOKIE_MAX_AGE_DAYS} * 24 * 60 * 60 * 1000));
        document.cookie = "{COOKIE_NAME}={value}; expires=" + d.toUTCString() + "; path=/; SameSite=Lax";
        </script>
        """,
        height=0,
    )


def _email_to_user_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def resolve_user(session_state) -> Optional[str]:
    """Return user_id if known this session, else None."""
    if "user_id" in session_state:
        return session_state["user_id"]

    device_token = _get_cookie()
    if device_token:
        user_id = get_user_id_for_device(device_token)
        if user_id:
            session_state["user_id"] = user_id
            return user_id

    return None


def show_auth_sidebar(session_state):
    """Render auth UI in sidebar. Sets session_state.user_id when resolved."""
    if session_state.get("user_id"):
        return

    user_id = resolve_user(session_state)
    if user_id:
        return

    with st.sidebar:
        st.markdown("### Sign in to Movie Buddy")
        st.markdown("Enter your email to save preferences across sessions.")

        email = st.text_input("Email address", key="auth_email")
        if st.button("Continue", key="auth_submit"):
            if "@" not in email:
                st.error("Please enter a valid email.")
                return
            user_id = _email_to_user_id(email)

            existing_token = _get_cookie()
            if existing_token and get_user_id_for_device(existing_token) == user_id:
                device_token = existing_token
            else:
                device_token = str(uuid.uuid4())
                register_device(device_token, user_id)
                _set_cookie(device_token)

            session_state["user_id"] = user_id
            st.rerun()
