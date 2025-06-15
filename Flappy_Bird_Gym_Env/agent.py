import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from replay_memory_buffer import ReplayMemoryBuffer
import yaml
import torch
import itertools
import os
import random


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
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)


        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_DQN = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemoryBuffer(max_size=self.replay_buffer_size)
            epsilon = self.epsilon_start

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            episode_reward = 0

            ## We will train indefinitely, and close the environment manually based on rewards per episode data.
            while not terminated:
                # Next action:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device) # Convert action to tensor
                else:
                    with torch.no_grad():
                        action = policy_DQN(state.unsqueeze(dim=0)).squeeze().argmax() # we do squeeze to make it 1D and we do item() to convert it back to integer from tensor

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                
                state = new_state

            rewards_per_episode.append(episode_reward)

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)


if __name__ == '__main__':
    agent = Agent(hyperparameter_set='cartpole1')
    agent.run(is_training=True, render=True) #Set render to true to see the agent play, and false to train faster