import flappy_bird_gymnasium
import gymnasium
from dqn import DQN

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    #We will use this for both training and testing
    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_DQN = DQN(num_states, num_actions).to(device)



env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()