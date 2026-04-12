"""
Study Planner OpenEnv - inference.py
FastAPI server implementing the OpenEnv spec: reset(), step(), state() API.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from openai import OpenAI


# ─── Models ───────────────────────────────────────────────────────────

class Subject(BaseModel):
    id: str
    name: str
    total_hours_needed: float
    hours_studied: float = 0.0
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    deadline_day: int = 7


class Task(BaseModel):
    id: str
    subject_id: str
    description: str
    estimated_hours: float
    completed: bool = False
    priority: Literal["low", "medium", "high"] = "medium"


class StudyPlannerState(BaseModel):
    episode_id: str
    day: int = 1
    max_days: int = 14
    subjects: List[Subject] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    total_hours_studied: float = 0.0
    energy_level: float = 1.0
    score: float = 0.0
    done: bool = False


class Action(BaseModel):
    action_type: Literal["study", "create_task", "complete_task", "rest"]
    payload: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"


class StepRequest(BaseModel):
    action: Action


# ─── Config ───────────────────────────────────────────────────────────

DIFFICULTY_CONFIGS = {
    "easy": [{"name": "Math", "total": 10}],
    "medium": [{"name": "Math", "total": 20}, {"name": "Physics", "total": 15}],
}


# ─── Core Functions ───────────────────────────────────────────────────

def _compute_reward(prev_progress, new_progress):
    reward = (new_progress - prev_progress) * 5
    return max(-1.0, min(1.0, reward))


def _initialize_state(difficulty):
    subjects = []
    tasks = []

    for s in DIFFICULTY_CONFIGS[difficulty]:
        sid = str(uuid.uuid4())
        subject = Subject(id=sid, name=s["name"], total_hours_needed=s["total"])
        subjects.append(subject)

    return StudyPlannerState(
        episode_id=str(uuid.uuid4()),
        subjects=subjects
    )


def _apply_action(state, action):
    payload = action.payload or {}

    if action.action_type == "study":
        subject = state.subjects[0]
        subject.hours_studied += 1
        state.total_hours_studied += 1
        state.day += 1
        return "studied"

    if action.action_type == "rest":
        state.energy_level = min(1.0, state.energy_level + 0.2)
        state.day += 1
        return "rested"

    return "ok"


def _check_done(state):
    total_needed = sum(s.total_hours_needed for s in state.subjects)
    total_done = sum(s.hours_studied for s in state.subjects)
    return total_done >= total_needed


# ─── FastAPI ──────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Optional[StudyPlannerState] = None


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    global _state
    difficulty = (req.difficulty if req else "medium")
    _state = _initialize_state(difficulty)
    return {"observation": _state}


@app.post("/step")
def step(req: StepRequest):
    global _state

    if _state is None:
        raise HTTPException(400, "call reset")

    prev = sum(s.hours_studied for s in _state.subjects)

    msg = _apply_action(_state, req.action)

    new = sum(s.hours_studied for s in _state.subjects)

    reward = _compute_reward(prev, new)
    _state.score += reward
    _state.done = _check_done(_state)

    return {
        "reward": reward,
        "done": _state.done,
        "info": {"msg": msg}
    }


@app.get("/state")
def state():
    return _state


# ─── LLM Agent Runner ─────────────────────────────────────────────────

if __name__ == "__main__":
    import urllib.request

    PORT = int(os.environ.get("PORT", 7860))
    BASE = f"http://localhost:{PORT}"

    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL"),
        api_key=os.environ.get("API_KEY"),
    )

    def post(path, data):
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req).read())

    # wait for server
    for _ in range(10):
        try:
            urllib.request.urlopen(BASE + "/healthz")
            break
        except:
            time.sleep(1)

    print("[START]")

    obs = post("/reset", {})

    for i in range(20):
        # LLM decides action
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "choose: study or rest"}],
        )

        action = "study" if "study" in resp.choices[0].message.content else "rest"

        step = post("/step", {"action": {"action_type": action}})

        print("[STEP]", step["reward"])

        if step["done"]:
            break

    print("[END]")
