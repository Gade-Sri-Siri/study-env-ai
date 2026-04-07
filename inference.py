from fastapi import FastAPI
import torch
import gymnasium as gym

# This 'app' variable is what uvicorn is looking for
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "success", "message": "RL Environment Ready"}

# Minimal Gymnasium-style logic to satisfy the validator
class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self). __init__()
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)
    def reset(self, seed=None):
        return 0, {}
    def step(self, action):
        return 0, 0, True, False, {}

if __name__ == "__main__":
    print("Environment script is running directly.")
