from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import TriageEnv
from models import Action, TaskLevel

# APP SETUP


app = FastAPI(
    title="Hospital Triage OpenEnv",
    description="""
    A real-world hospital emergency triage environment
    where an AI agent learns to prioritize patients
    under dynamic conditions and limited resources.
    
    Built for Meta x Scaler OpenEnv Hackathon.
    """,
    version="1.0.0",
    docs_url="/docs",
)

# allow all origins — needed for HF Space
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# single global env instance
env = TriageEnv()

# REQUEST MODELS
# what the API expects to receive


class ResetRequest(BaseModel):
    task_level: Optional[str] = "easy"

class StepRequest(BaseModel):
    patient_id: str
    priority: str
    ward: str
    treatment: str
    reasoning: Optional[str] = None

# ENDPOINTS

@app.get("/")
def root():
    """
    Health check endpoint.
    HF Space pings this to verify server is alive.
    Must return 200.
    """
    return {
        "status": "ok",
        "environment": "hospital-triage-openenv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/docs"]
    }

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Returns first Observation.
    
    Args:
        task_level: easy / medium / hard
    """
    try:
        # validate task level
        valid_levels = ["easy", "medium", "hard"]
        if request.task_level not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"task_level must be one of {valid_levels}"
            )

        observation = env.reset(request.task_level)

        return {
            "status": "ok",
            "task_level": request.task_level,
            "observation": observation.dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )

@app.post("/step")
def step(request: StepRequest):
    """
    Submit agent action and get next observation.
    Returns observation + reward + done + info.
    
    Args:
        patient_id: ID of patient being triaged
        priority:   immediate/urgent/non_urgent/deceased
        ward:       ICU/emergency/general/waiting
        treatment:  cardiac_protocol/trauma_protocol/
                    respiratory_protocol/basic_care/observe_only
        reasoning:  optional agent reasoning text
    """
    try:
        if env.done:
            raise HTTPException(
                status_code=400,
                detail="Episode is done. Call /reset first."
            )

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
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Step failed: {str(e)}"
        )

@app.get("/state")
def state():
    """
    Get full current environment state.
    Can be called anytime during episode.
    Returns complete snapshot of env.
    """
    try:
        current_state = env.state()
        return {
            "status": "ok",
            "state": current_state.dict(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"State failed: {str(e)}"
        )

# RUN SERVER

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
    )