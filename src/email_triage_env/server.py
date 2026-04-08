from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import EmailTriageEnv
from .models import EmailTriageAction, EmailTriageState, EmailTriageStepResult


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for OpenEnv Round 1.",
    version="0.1.0",
)
_env = EmailTriageEnv()


@app.get("/")
async def root() -> dict:
    return {
        "name": "email-triage-openenv",
        "status": "ok",
        "docs": "/docs",
    }


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
            "email_count": len(task.emails),
        }
        for task in _env.tasks()
    ]


@app.post("/reset", response_model=EmailTriageStepResult)
async def reset(payload: Optional[ResetRequest] = None) -> EmailTriageStepResult:
    task_name = payload.task_name if payload else None
    try:
        return await _env.reset(task_name=task_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=EmailTriageStepResult)
async def step(action: EmailTriageAction) -> EmailTriageStepResult:
    try:
        return await _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EmailTriageState)
async def state() -> EmailTriageState:
    try:
        return await _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/close")
async def close() -> dict:
    await _env.close()
    return {"status": "closed"}
