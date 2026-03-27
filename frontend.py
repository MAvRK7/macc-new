import streamlit as st
import requests
import time

# ------------------ Config ------------------
BASE_URL = st.secrets["api"]["BASE_URL"]   # Must be set in Streamlit secrets

# ------------------ Helpers ------------------
def start_project(spec: str, github_repo: str = "") -> str | None:
    payload = {"spec": spec, "github_repo": github_repo}
    try:
        resp = requests.post(f"{BASE_URL}/generate-project", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["session_id"]
    except Exception as e:
        st.error(f"❌ Failed to start project: {e}")
        return None

def poll_updates(session_id: str):
    """Poll backend and update UI in real-time"""
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

            for msg in data.get("messages", []):
                typ = msg.get("type")
                content = msg.get("message", "")

                if typ == "status":
                    status_container.write(f"📢 **{content}**")
                elif typ == "code":
                    code_text += content + "\n"
                    code_container.text_area(
                        "Generated Code", 
                        value=code_text, 
                        height=500, 
                        key=f"code_{len(code_text)}"
                    )
        except Exception as e:
            st.error(f"❌ Polling error: {e}")
            break

        time.sleep(0.6)   # Slightly longer sleep to reduce load

    return code_text, repo_url

def apply_suggestion(session_id: str, suggestion: str):
    try:
        resp = requests.post(
            f"{BASE_URL}/suggest-changes", 
            json={"session_id": session_id, "suggestion": suggestion}, 
            timeout=10
        )
        resp.raise_for_status()
        st.success("✅ Suggestion submitted! Updating code...")
        return poll_updates(session_id)
    except Exception as e:
        st.error(f"❌ Failed to apply suggestion: {e}")
        return None, None

def commit_to_github(session_id: str):
    try:
        resp = requests.post(f"{BASE_URL}/commit", json={"session_id": session_id}, timeout=15)
        resp.raise_for_status()
        url = resp.json().get("repo_url")
        if url:
            st.success(f"✅ Code committed successfully! [Open Repository]({url})")
        return url
    except Exception as e:
        st.error(f"❌ GitHub commit failed: {e}")
        return None

# ------------------ Main UI ------------------
st.title("🛠 MACC - Multi-Agent AI Code Collaborator")
st.markdown("**LangGraph-powered** Python project generator")

default_prompt = "Build a Python CLI for weather forecasting with email alerts"

spec = st.text_area("Project Specification", value=default_prompt, height=100)
github_repo = st.text_input("GitHub Repo (optional - leave blank for auto-generated)", value="")

if st.button("🚀 Generate Project", type="primary"):
    if not spec.strip():
        st.warning("Please enter a project specification")
    else:
        with st.spinner("Starting multi-agent workflow..."):
            session_id = start_project(spec, github_repo)
            if session_id:
                st.session_state.session_id = session_id
                st.session_state.code, st.session_state.repo_url = poll_updates(session_id)

# Show results
if st.session_state.get("code"):
    st.subheader("✅ Generated Code")
    st.code(st.session_state.code, language="python")

if st.session_state.get("repo_url"):
    st.subheader("📂 GitHub Repository")
    st.markdown(f"[🔗 Open on GitHub]({st.session_state.repo_url})")

# Suggestion section
st.subheader("💡 Refine with Suggestion")
suggestion = st.text_area("Enter your suggestion to improve the code:", height=80)

if st.button("Apply Suggestion") and suggestion.strip():
    if not st.session_state.get("session_id"):
        st.warning("Please generate a project first")
    else:
        st.session_state.code, st.session_state.repo_url = apply_suggestion(
            st.session_state.session_id, suggestion
        )

# Commit button
if st.session_state.get("session_id") and st.session_state.get("code"):
    if st.button("💾 Commit to GitHub"):
        commit_to_github(st.session_state.session_id)
