from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Hospital Triage OpenEnv")

# TODO: import env after models.py and env.py are ready
# from env import TriageEnv
# env = TriageEnv()

@app.get("/")
def root():
    return {"message": "Hospital Triage OpenEnv is running"}

@app.post("/reset")
def reset():
    # TODO: return env.reset()
    return {"status": "ok", "message": "reset called"}

@app.post("/step")
def step(action: dict):
    # TODO: return env.step(action)
    return {"status": "ok", "message": "step called"}

@app.get("/state")
def state():
    # TODO: return env.state()
    return {"status": "ok", "message": "state called"}
