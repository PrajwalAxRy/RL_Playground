import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from replay_memory_buffer import ReplayMemoryBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    #We will use this for both training and testing
    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_DQN = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemoryBuffer(max_size=10000)

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0

            ## We will train indefinitely, and close the environment manually based on rewards per episode data.
            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()      #TODO: Currently taking actions ramdomly, fix this with epsilon-greedy.

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)
                
                episode_reward += reward

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                
                state = new_state

            rewards_per_episode.append(episode_reward)