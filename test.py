# patch for NumPy ≥1.24 (restore the old alias)
import numpy as np
np.bool8 = np.bool_

import gymnasium as gym

def run_random_episode(env_id="LunarLander-v2"):
    # request human rendering up front
    env = gym.make(env_id, render_mode="human")
    
    obs, info = env.reset()        # now returns (obs, info)
    done = False
    total_reward = 0.0

    while not done:
        # pick a random action
        action = env.action_space.sample()
        
        # new API: step() → obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # episode is over when either terminated or truncated
        done = terminated or truncated

    print(f"Episode finished — total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    run_random_episode()
