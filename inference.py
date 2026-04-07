from environment import StudyEnv
import random

levels = ["easy", "medium", "hard"]

for level in levels:
    print(f"\nRunning Level: {level}")
    
    env = StudyEnv(level=level)
    state = env.reset()
    
    done = False
    total_reward = 0
    
    actions = ["Math", "Python", "Physics", "DSA", "English", "AI", "Break"]
    
    while not done:
        action = random.choice(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    
    print("Final State:", state)
    print("Total Reward:", total_reward)