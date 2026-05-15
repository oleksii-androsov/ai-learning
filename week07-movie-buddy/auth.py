import hashlib
import uuid
from typing import Optional
import streamlit as st
from streamlit_cookies_controller import CookieController
from memory import get_user_id_for_device, register_device

COOKIE_NAME = "mb_device_token"

# One controller instance per session
def _controller() -> CookieController:
    if "_cookie_controller" not in st.session_state:
        st.session_state._cookie_controller = CookieController()
    return st.session_state._cookie_controller


def _email_to_user_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def resolve_user(session_state) -> Optional[str]:
    """Return user_id if known this session, else None."""
    if "user_id" in session_state:
        return session_state["user_id"]

    # Try reading device token from cookie
    controller = _controller()
    device_token = controller.get(COOKIE_NAME)
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

    # Try cookie first (may succeed on rerun after controller initialises)
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

            # Reuse existing device token for this browser if one exists
            controller = _controller()
            existing_token = controller.get(COOKIE_NAME)
            if existing_token and get_user_id_for_device(existing_token) == user_id:
                device_token = existing_token
            else:
                device_token = str(uuid.uuid4())
                register_device(device_token, user_id)
                controller.set(COOKIE_NAME, device_token, max_age=365 * 24 * 3600)

            session_state["user_id"] = user_id
            st.rerun()
