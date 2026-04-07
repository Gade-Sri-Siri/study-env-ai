from fastapi import FastAPI
import torch
import gymnasium as gym

app = FastAPI()

# Basic RL Environment Setup for Validation
@app.get("/")
def read_root():
    return {"status": "Environment Active", "framework": "PyTorch", "device": str(torch.cuda.is_available())}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Placeholder for the actual RL inference logic
def run_inference():
    # This ensures the script is runnable via 'python3 inference.py' as well
    print("Inference engine initialized.")

if __name__ == "__main__":
    run_inference()
