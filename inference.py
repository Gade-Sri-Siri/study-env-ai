from fastapi import FastAPI
from environment import StudyEnv

app = FastAPI()

env = StudyEnv()

@app.post("/reset")
def reset_env():
    state = env.reset()
    return {
        "state": state
    }

@app.post("/step")
def step_env(action: str):
    state, reward, done = env.step(action)
    return {
        "state": state,
        "reward": reward,
        "done": done
    }

@app.get("/")
def home():
    return {"message": "Study Env API is running"}
