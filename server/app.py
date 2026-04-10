"""
server/app.py — OpenEnv-compliant entry point for Hospital Triage environment.
This file is referenced by pyproject.toml [project.scripts] as server = "server.app:main"
"""

import os
import uvicorn

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}")

# Dual-import pattern required by OpenEnv
try:
    from ..env import TriageEnv
    from ..models import Action
except ImportError:
    try:
        from env import TriageEnv
        from models import Action
    except ImportError:
        TriageEnv = None
        Action = None

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hospital Triage OpenEnv",
    description="Hospital emergency triage environment for Meta x Scaler OpenEnv Hackathon.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mutable env container (so /step always sees latest env)
env_container = {"env": TriageEnv() if TriageEnv else None}


def get_env():
    return env_container["env"]


# ── Request models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: Optional[str] = "easy"


class StepRequest(BaseModel):
    patient_id: str
    priority: str
    ward: str
    treatment: str
    reasoning: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "hospital-triage-openenv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "hospital-triage-openenv"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    try:
        valid_levels = ["easy", "medium", "hard"]
        if request.task_level not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"task_level must be one of {valid_levels}",
            )
        env_container["env"] = TriageEnv()
        observation = env_container["env"].reset(request.task_level)
        return {
            "status": "ok",
            "task_level": request.task_level,
            "observation": observation.dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(request: StepRequest):
    try:
        env = get_env()
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized.")
        if env.done:
            raise HTTPException(status_code=400, detail="Episode done. Call /reset first.")

        action = Action(
            patient_id=request.patient_id,
            priority=request.priority,
            ward=request.ward,
            treatment=request.treatment,
            reasoning=request.reasoning,
        )
        result = env.step(action)
        return {
            "status": "ok",
            "observation": result.observation.dict(),
            "reward": result.reward.dict(),
            "done": result.done,
            "info": result.info,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
def state():
    try:
        current_state = get_env().state()
        return {"status": "ok", "state": current_state.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")


# ── main() — required by [project.scripts] ────────────────────────────────

def main():
    """Entry point called by `server` CLI command (pyproject.toml scripts)."""
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()