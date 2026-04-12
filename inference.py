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
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# ─── Domain Models ────────────────────────────────────────────────────────────

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

class StudySession(BaseModel):
    task_id: str
    hours: float
    quality: Literal["poor", "average", "good", "excellent"] = "average"

# ─── Environment State ─────────────────────────────────────────────────────────

class StudyPlannerState(BaseModel):
    episode_id: str
    day: int = 1
    max_days: int = 14
    subjects: List[Subject] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    completed_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    total_hours_studied: float = 0.0
    energy_level: float = 1.0
    score: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

# ─── Action Models ────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: Literal["study", "create_task", "complete_task", "rest", "review_schedule"]
    payload: Optional[Dict[str, Any]] = None

class ResetRequest(BaseModel):
    difficulty: Optional[Literal["easy", "medium", "hard"]] = "medium"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: Action

# ─── Response Models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    day: int
    max_days: int
    energy_level: float
    subjects: List[Dict[str, Any]]
    pending_tasks: List[Dict[str, Any]]
    completed_tasks_count: int
    total_hours_studied: float
    overall_progress: float
    days_remaining: int
    urgent_subjects: List[str]

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any]

# ─── Grade Configurations ─────────────────────────────────────────────────────

DIFFICULTY_CONFIGS = {
    "easy": {
        "subjects": [
            {"name": "Mathematics", "total_hours": 10.0, "difficulty": "easy", "deadline_day": 14},
            {"name": "English Literature", "total_hours": 8.0, "difficulty": "easy", "deadline_day": 14},
            {"name": "History", "total_hours": 6.0, "difficulty": "easy", "deadline_day": 12},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 2,
    },
    "medium": {
        "subjects": [
            {"name": "Mathematics", "total_hours": 20.0, "difficulty": "hard", "deadline_day": 10},
            {"name": "Physics", "total_hours": 15.0, "difficulty": "medium", "deadline_day": 12},
            {"name": "Chemistry", "total_hours": 12.0, "difficulty": "medium", "deadline_day": 14},
            {"name": "English", "total_hours": 8.0, "difficulty": "easy", "deadline_day": 14},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 3,
    },
    "hard": {
        "subjects": [
            {"name": "Advanced Mathematics", "total_hours": 30.0, "difficulty": "hard", "deadline_day": 8},
            {"name": "Quantum Physics", "total_hours": 25.0, "difficulty": "hard", "deadline_day": 10},
            {"name": "Organic Chemistry", "total_hours": 20.0, "difficulty": "hard", "deadline_day": 9},
            {"name": "Computer Science", "total_hours": 18.0, "difficulty": "medium", "deadline_day": 12},
            {"name": "Statistics", "total_hours": 15.0, "difficulty": "medium", "deadline_day": 11},
        ],
        "max_days": 14,
        "initial_tasks_per_subject": 4,
    },
}

TASK_TEMPLATES = {
    "Mathematics": ["Solve practice problems", "Review theorems", "Complete exercises", "Study formulas", "Work on proofs"],
    "Physics": ["Study concepts", "Solve numerical problems", "Review lab notes", "Practice derivations", "Read textbook chapter"],
    "Chemistry": ["Learn reactions", "Practice equations", "Review periodic table", "Study mechanisms", "Complete worksheets"],
    "English Literature": ["Read assigned texts", "Write essay outline", "Analyze themes", "Review grammar rules", "Prepare notes"],
    "History": ["Review timeline", "Study key events", "Analyze primary sources", "Make flashcards", "Write summ
