import streamlit as st
import requests
import time

# ------------------ Config ------------------
BASE_URL = st.secrets["api"]["BASE_URL"]  # e.g., "https://macc-project-n5v3.onrender.com"

# ------------------ Helpers ------------------
def start_project(spec: str, github_repo: str = "") -> str:
    """Send request to backend to start project generation; returns session_id"""
    payload = {"spec": spec, "github_repo": github_repo}
    try:
        resp = requests.post(f"{BASE_URL}/generate-project", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to start project: {e}")
        return None

def poll_updates(session_id: str):
    """Poll updates from backend until done; update Streamlit UI"""
    status_container = st.empty()
    code_container = st.empty()
    code_text = ""

    done = False
    repo_url = None
    while not done:
        try:
            resp = requests.get(f"{BASE_URL}/updates/{session_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            done = data.get("done", False)
            repo_url = data.get("repo_url")
            messages = data.get("messages", [])
            for msg in messages:
                typ = msg.get("type")
                content = msg.get("message")
                if typ == "status":
                    status_container.text(f"📢 {content}")
                elif typ == "code":
                    code_text += content + "\n"
                    code_container.text_area("Generated Code", value=code_text, height=400, key=f"code_{len(code_text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error polling updates: {e}")
            break
        time.sleep(0.5)

    return code_text, repo_url

def apply_suggestion(session_id: str, suggestion: str):
    """Send a suggestion to backend for refinement"""
    try:
        resp = requests.post(f"{BASE_URL}/suggest-changes", json={"session_id": session_id, "suggestion": suggestion}, timeout=10)
        resp.raise_for_status()
        st.success("✅ Suggestion submitted! Polling for updates...")
        return poll_updates(session_id)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to apply suggestion: {e}")
        return None, None

def commit_to_github(session_id: str):
    """Commit code to GitHub via backend"""
    try:
        resp = requests.post(f"{BASE_URL}/commit", json={"session_id": session_id}, timeout=10)
        resp.raise_for_status()
        url = resp.json().get("repo_url")
        if url:
            st.success(f"✅ Code committed to GitHub: [Open Repo]({url})")
        return url
    except requests.exceptions.RequestException as e:
        st.error(f"❌ GitHub commit failed: {e}")
        return None

# ------------------ UI ------------------
st.title("🛠 MACC - Multi-Agent AI Code Collaborator")
st.write("Generate Python projects with AI agents. Enter your project specification below:")

default_prompt = "Build a Python CLI for weather forecasting with email alerts"
spec = st.text_area("Project Specification", value=default_prompt, height=80)
github_repo = st.text_input("GitHub Repo (optional)", value="")

if st.button("Generate Project"):
    session_id = start_project(spec, github_repo)
    if session_id:
        st.session_state.session_id = session_id
        st.session_state.code, st.session_state.repo_url = poll_updates(session_id)

# ------------------ Show Results ------------------
if st.session_state.get("code"):
    st.subheader("✅ Generated Code")
    st.code(st.session_state.code, language="python")

if st.session_state.get("repo_url"):
    st.subheader("📂 GitHub Repository")
    st.markdown(f"[Open on GitHub]({st.session_state.repo_url})")

# ------------------ Suggestions ------------------
st.subheader("💡 Refine / Suggest Changes")
suggestion = st.text_area("Enter your suggestion to improve the code:")
if st.button("Apply Suggestion") and suggestion.strip():
    if not st.session_state.get("session_id"):
        st.warning("Generate a project first before applying suggestions.")
    else:
        st.session_state.code, st.session_state.repo_url = apply_suggestion(st.session_state.session_id, suggestion)

# ------------------ Commit ------------------
if st.session_state.get("session_id") and st.session_state.get("code"):
    if st.button("Commit to GitHub"):
        commit_to_github(st.session_state.session_id)
