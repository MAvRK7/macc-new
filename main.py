# main.py
import os
import warnings
import asyncio
import logging
import time
import random
import json
import signal
import uuid
import concurrent.futures
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Any, Optional, TypedDict
from openai import OpenAI
# from mistralai.client import Mistral
from mistralai import Mistral

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from github import Github

#LangGraph
from langgraph.graph import StateGraph, START, END

# ---------------- Logging & env ----------------
logging.basicConfig(level=logging.INFO, filename="agent_logs.txt")
logging.info("Starting MACC application")

os.environ["PYDANTIC_SKIP_VALIDATING_ASSIGNMENT"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in environment")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN missing in environment")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY missing in environment")

mistral = Mistral(api_key=MISTRAL_API_KEY)

# ---------------- FastAPI ----------------
app = FastAPI(title="MACC - Multi-Agent AI Code Collaborator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory state ----------------
# session_id -> list[dict] of messages (message consumed on GET /updates)
session_messages: Dict[str, list] = {}
# session_id -> final context (spec, code, readme, repo name/url)
project_context: Dict[str, Dict[str, Any]] = {}
# sentinel for finished sessions
session_done: Dict[str, bool] = {}

# executor for blocking Crew calls
executor = ThreadPoolExecutor(max_workers=4)

# ---------------- Models ----------------
class ProjectRequest(BaseModel):
    spec: str
    github_repo: Optional[str] = ""

class SuggestionRequest(BaseModel):
    session_id: str
    suggestion: str

class CommitRequest(BaseModel):
    session_id: str

# ---------------- Utility helpers ----------------
def safe_slug(text: str, max_len: int = 28) -> str:
    """Create a safe repo name from the spec (lowercase, alnum + hyphen)."""
    s = text.lower()
    # keep alnum and spaces
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    s = "-".join(s.split())
    s = s.strip("-")
    if not s:
        s = "macc-project"
    return (s[:max_len]).rstrip("-")

def ensure_session(session_id: str):
    if session_id not in session_messages:
        session_messages[session_id] = []
    if session_id not in session_done:
        session_done[session_id] = False
    if session_id not in project_context:
        project_context[session_id] = {}

def enqueue_message(session_id: str, typ: str, message: str):
    """Append a message to session message list (for polling)."""
    ensure_session(session_id)
    session_messages[session_id].append({"type": typ, "message": message})

def drain_messages(session_id: str) -> list:
    """Return accumulated messages and clear them."""
    ensure_session(session_id)
    msgs = session_messages[session_id][:]
    session_messages[session_id] = []
    return msgs
# ---------------- LangGraph State ----------------
class GraphState(TypedDict):
    spec: str
    tasks: str
    code: str
    refined_code: str

# ---------------- Multi LLM ----------------

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "MACC",
    },
)

class MultiLLM:
    def __init__(self, fallback_model="qwen/qwen3-coder:free", timeout=20):
        self.fallback_model = fallback_model
        self.timeout = timeout

    # -------- Normalize messages --------
    def _normalize(self, prompt:str):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt
    
    # -------- Cache --------
    @lru_cache(maxsize=128)
    def _cached_call(self, messages_json: str) -> str:
        """Cached core call with fallback logic"""
        messages = json.loads(messages_json)
        
        # ---------------- PRIMARY: Mistral ----------------
        for attempt in range(3):
            try:
                res = mistral.chat.complete(
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=0.2,
                )
                content = res.choices[0].message.content
                if content and content.strip():
                    logging.info("Used Mistral (primary)")
                    return content.strip()
                else:
                    logging.warning(f"Empty response from Mistral on attempt {attempt+1}")
            except Exception as e:
                logging.error(f"Mistral failed: {e}")
                time.sleep(0.8)

        # ---------------- FALLBACK: OpenRouter ----------------
        try:
            logging.info("Falling back to OpenRouter...")
            response = openrouter.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4000,          # prevent overly long responses
            )
            content = response.choices[0].message.content
            if content and content.strip():   # basic quality check
                logging.info("Used OpenRouter (fallback)")
                return content.strip()  
        except Exception as e:
            logging.error(f"OpenRouter fallback failed: {e}")
            return "# Error: Both LLM providers failed. Please try again."
    
    # -------- REQUIRED method -------- 
    def call(self, prompt: str) -> str:
        """Public method - uses cache + timeout protection"""
        messages = self._normalize(prompt)
        messages_json = json.dumps(messages, sort_keys=True)   # make cacheable

        future = executor.submit(self._cached_call, messages_json)

        # Run in thread with timeout to prevent hanging

        try:
            return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            logging.error("LLM call timed out")
            return "# Error: LLM call timed out. Please try again."
        except Exception as e:
            logging.error(f"Unexpected error in LLM call: {e}")
            return "# Error: LLM call failed unexpectedly."

llm = MultiLLM(fallback_model="qwen/qwen3-coder:free")

# ---------------- LangGraph Nodes ----------------
#---PLANNER NODE---
def planner_node(state: GraphState) -> dict:
    prompt = f"""Break down this project specification into a clear numbered list of tasks.
Project: {state['spec']}
Output ONLY the numbered list. No extra text."""
    tasks = llm.call(prompt)
    return {"tasks": tasks}

#--CODER NODE--
def coder_node(state: GraphState) -> dict:
    prompt = f"""Write a complete, production-ready, SINGLE-FILE Python script for the following project.

Project Specification:
{state['spec']}

Tasks:
{state.get('tasks', '')}

Rules:
- All code in ONE file
- Proper imports at the top
- Good error handling
- Include if __name__ == '__main__': block
- Output ONLY the full Python code. No explanations, no markdown fences.
- If the spec is too complex, write a simplified version that captures the core idea."""
    code = llm.call(prompt)
    return {"code": code}

#--REVIEWER NODE--
def reviewer_node(state: GraphState) -> dict:
    prompt = f"""Review and improve the following Python code.

Current code:
{state['code']}

Improvements needed:
- Fix any bugs
- Improve structure and readability
- Add better error handling where missing
- Keep it as a single file

Output ONLY the full improved Python code. No explanations."""
    refined = llm.call(prompt)
    return {"refined_code": refined}

# Build the graph once
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("planner", planner_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "reviewer")
    builder.add_edge("reviewer", END)
    
    return builder.compile()

graph = build_graph()

# ---------------- GitHub tool ----------------

class GitHubTool:
    def push(self, repo_name: str, code: str, filename="main.py", readme=None):
        g = Github(GITHUB_TOKEN)
        user = g.get_user()
        repo_short = repo_name.split("/")[-1] if "/" in repo_name else repo_name
        
        try:
            repo = user.get_repo(repo_short)
        except Exception:
            repo = user.create_repo(repo_short, auto_init=True)
        
        # Update or create main file
        try:
            repo.create_file(filename, "Add main code", code)
        except Exception:
            try:
                existing = repo.get_contents(filename)
                repo.update_file(filename, "Update main code", code, existing.sha)
            except Exception:
                pass
        
        if readme:
            try:
                repo.create_file("README.md", "Add README", readme)
            except Exception:
                try:
                    existing = repo.get_contents("README.md")
                    repo.update_file("README.md", "Update README", readme, existing.sha)
                except Exception:
                    pass
        
        return f"https://github.com/{user.login}/{repo_short}"

github_tool = GitHubTool()

# ---------------- Background tasks ----------------
async def generate_background(session_id: str, spec: str, github_repo: Optional[str]):
    ensure_session(session_id)
    try:
        enqueue_message(session_id, "status", "Starting project generation...")
        if not spec or len(spec.strip()) < 3:
            enqueue_message(session_id, "status", "Error: Project spec too short.")
            session_done[session_id] = True
            return

        # derive repo name from prompt if not provided
        if github_repo and github_repo.strip():
            repo = github_repo.strip()
            enqueue_message(session_id, "status", f"Using provided GitHub repo: {repo}")
        else:
            slug = safe_slug(spec)
            # ensure uniqueness by appending short UUID
            repo = f"{slug}-{uuid.uuid4().hex[:6]}"
            # store repo as username/repo? We'll let GitHubTool use user's login later.
            enqueue_message(session_id, "status", f"Auto-generated repo name: {repo}")

        # PLANNER (run in thread)
        enqueue_message(session_id, "status", "Planner: breaking down tasks...")
        enqueue_message(session_id, "status", "Coder generating code...")
        enqueue_message(session_id, "status", "Reviewer improving code...")

        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                executor, lambda: graph.invoke({"spec": spec})
            ),
            timeout=90
        )

        tasks = result.get("tasks", "")
        code = result.get("refined_code") or result.get("code", "")
        
        if not code or len(code.strip()) < 50:
            code = "# Error: LLM returned empty output. Please try a simpler specification."

        # Stream the final code
        for line in code.splitlines():
            enqueue_message(session_id, "code", line)

        # Create README
        readme = f"""# {repo}
## Description
{spec}

## Generated Tasks
{tasks}

Generated by MACC (LangGraph)
```bash
# To run:
python main.py
```"""
        project_context[session_id] = {
            "spec": spec,
            "github_repo": repo,
            "tasks": tasks,
            "code": code,
            "readme": readme,
            "repo_url": None,
        }

        enqueue_message(session_id, "status", f"Project ready! Repo: {repo}")
        session_done[session_id] = True

    except Exception as e:
        logging.exception("generate_background error")
        enqueue_message(session_id, "status", f"Error: {str(e)}")
        session_done[session_id] = True


async def refine_background(session_id: str, suggestion: str):
    ensure_session(session_id)
    try:
        if session_id not in project_context:
            enqueue_message(session_id, "status", "Error: session not found for refinement.")
            session_done[session_id] = True
            return
        ctx = project_context[session_id]
        current_code = ctx.get("code", "")
        enqueue_message(session_id, "status", f"Applying suggestion: {suggestion}")
        prompt = f"""Refine the following code based on the user suggestion.
Suggestion: {suggestion}
Current Code:
{current_code}

Output ONLY the full refined Python code. No explanations, no markdown."""
        refined = llm.call(prompt)
        
        if not refined or len(refined.strip()) < 50:
            refined = current_code
            enqueue_message(session_id, "status", "LLM returned empty refinement, keeping original code.")
        else:
            ctx["code"]=refined #update

        # stream refined code
        for ln in refined.splitlines():
            enqueue_message(session_id, "code", ln)
        enqueue_message(session_id, "status", "Refinement applied.")
        session_done[session_id] = True
    except Exception as e:
        logging.exception("refine_background error")
        enqueue_message(session_id, "status", f"Unhandled error: {e}")
        session_done[session_id] = True

        
# ---------------- API endpoints ----------------

@app.post("/generate-project")
async def generate_project(req: ProjectRequest):
    """Start generation in background and immediately return session_id."""
    session_id = str(uuid.uuid4())
    # initialize
    ensure_session(session_id)
    session_done[session_id] = False
    # start background
    asyncio.create_task(generate_background(session_id, req.spec, req.github_repo))
    return {"session_id": session_id}

@app.get("/updates/{session_id}")
async def get_updates(session_id: str):
    """Return queued messages (and done flag) for session; clears returned messages."""
    ensure_session(session_id)
    msgs = drain_messages(session_id)
    done = session_done.get(session_id, False)
    # include repo_url if available
    repo_url = project_context.get(session_id, {}).get("repo_url")
    return {"messages": msgs, "done": done, "repo_url": repo_url}

@app.post("/suggest-changes")
async def suggest_changes(req: SuggestionRequest):
    """Start refinement (background) and return session_id (same)."""
    if req.session_id not in session_messages and req.session_id not in project_context:
        raise HTTPException(status_code=404, detail="Session not found")
    ensure_session(req.session_id)
    session_done[req.session_id] = False
    asyncio.create_task(refine_background(req.session_id, req.suggestion))
    return {"session_id": req.session_id}

@app.post("/commit")
async def commit(req: CommitRequest):
    sid = req.session_id
    if sid not in project_context:
        raise HTTPException(status_code=404, detail="Session not found")
    ctx = project_context[sid]
    repo_name = ctx.get("github_repo")
    code = ctx.get("code", "")
    readme = ctx.get("readme", "")
    if not repo_name:
        raise HTTPException(status_code=400, detail="No repo name in session")
    try:
        url = github_tool.push(repo_name, code, filename="main.py", readme=readme)
        ctx["repo_url"] = url
        enqueue_message(sid, "status", f"Code committed to GitHub: {url}")
        return {"status": "committed", "repo_url": url}
    except Exception as e:
        logging.exception("commit failed")
        raise HTTPException(status_code=500, detail=f"GitHub commit failed: {e}")

@app.get("/")
async def root():
    return JSONResponse({"message": "MACC API running - all good"})

# ---------------- Run if main ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
