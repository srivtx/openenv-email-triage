from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .environment import PersonalAssistantConflictEnv
from .models import ConflictAction, ConflictState, ConflictStepResult


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


app = FastAPI(
    title="Personal Assistant Conflict Resolver v2",
    description="Long-horizon personalized scheduling and delegation OpenEnv environment.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = PersonalAssistantConflictEnv()
_static_dir = Path(__file__).parent / "static"

if _static_dir.is_dir():
    # Serves /static/...  (local + hf.space direct)
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    # HF Spaces with base_path: /web  resolves assets under /web/static/...
    app.mount("/web/static", StaticFiles(directory=str(_static_dir)), name="static_web")


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def root():
    index = _static_dir / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Conflict Resolver v2</h1><p>API is running. Visit /docs</p>")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
async def tasks() -> list[dict]:
    return [
        {
            "id": task.id,
            "title": task.title,
            "difficulty": task.difficulty,
            "description": task.description,
            "max_steps": task.max_steps,
            "conflict_count": len(task.conflicts),
        }
        for task in _env.tasks()
    ]


@app.post("/reset", response_model=ConflictStepResult)
async def reset(payload: Optional[ResetRequest] = None) -> ConflictStepResult:
    task_name = payload.task_name if payload else None
    try:
        return await _env.reset(task_name=task_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=ConflictStepResult)
async def step(action: ConflictAction) -> ConflictStepResult:
    try:
        return await _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=ConflictState)
async def state() -> ConflictState:
    try:
        return await _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/close")
async def close() -> dict:
    await _env.close()
    return {"status": "closed"}
