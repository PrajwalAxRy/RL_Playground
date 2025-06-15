import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from replay_memory_buffer import ReplayMemoryBuffer
import yaml
import torch
from torch import nn
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
        network_sync_rate: 20
        learning_rate: 0.001
        discount_factor: 0.99
        """  
        self.replay_buffer_size = config['replay_buffer_size']
        self.mini_batch_size = config['mini_batch_size']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.network_sync_rate = config['network_sync_rate']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam

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

            target_DQN = DQN(num_states, num_actions).to(device)
            target_DQN.load_state_dict(policy_DQN.state_dict())

            #We well use this to update the target network
            step_counter = 0

            self.optimizer = torch.optim.Adam(policy_DQN.parameters(), lr=self.learning_rate)

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

                    step_counter += 1
                
                state = new_state

            rewards_per_episode.append(episode_reward)

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if len(memory) > self.mini_batch_size and is_training:
                # Sample a mini-batch from the memory
                mini_batch = memory.sample(self.mini_batch_size)
                
                self.optimize(policy_DQN, target_DQN, mini_batch) ## This is where training happens

                if step_counter > self.network_update_frequency:
                    target_DQN.load_state_dict(policy_DQN.state_dict())
                    step_counter = 0

    def optimize(self, policy_DQN, target_DQN, mini_batch):

        for state, action, reward, new_state, terminated in mini_batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor * target_DQN(new_state.unsqueeze(dim=0)).squeeze().max()

            current_q = policy_DQN(state.unsqueeze(dim=0)).squeeze()

            loss = self.loss_fn(current_q[action], target)

            #Optimize the model
            self.optimizer.zero_grad() # Clear the gradients
            loss.backward() # compute the gradients
            self.optimizer.step() # Update the model parameters

        


if __name__ == '__main__':
    agent = Agent(hyperparameter_set='cartpole1')
    agent.run(is_training=True, render=True) #Set render to true to see the agent play, and false to train faster