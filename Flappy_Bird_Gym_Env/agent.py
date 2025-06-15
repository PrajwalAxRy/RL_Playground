import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from replay_memory_buffer import ReplayMemory
import yaml
import torch
import itertools
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def __init__(self, config_path='.\Flappy_Bird_Gym_Env\hyperparameters.yaml', hyperparameter_set=None):
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            config = config_dict[hyperparameter_set]

        # Config details
        """
        replay_buffer_size: 10000
        mini_batch_size: 32
        epsilon_start: 0.9
        epsilon_min: 0.05
        epsilon_decay: 0.9995
        """  
        self.replay_buffer_size = config['replay_buffer_size']
        self.mini_batch_size = config['mini_batch_size']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']   
        
    #We will use this for both training and testing
    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_DQN = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemoryBuffer(max_size=self.replay_buffer_size)

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



if __name__ == '__main__':
    agent = Agent()
    agent.run(render=True) #Set render to true to see the agent play, and false to train faster