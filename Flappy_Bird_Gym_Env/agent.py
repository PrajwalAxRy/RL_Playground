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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Define the path for run-related files relative to the script execution directory
RUNS_DIR_PATH = os.path.join("Flappy_Bird_Gym_Env", "RUNS_DIR")
os.makedirs(RUNS_DIR_PATH, exist_ok=True)

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib


device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

    def __init__(self, config_path='./Flappy_Bird_Gym_Env/hyperparameters.yaml', hyperparameter_set=None):
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            config = config_dict[hyperparameter_set]

        # Config details
        """
        env_name: FlappyBird-v0
        replay_buffer_size: 10000
        mini_batch_size: 32
        epsilon_start: 0.9
        epsilon_min: 0.05
        epsilon_decay: 0.9995
        network_sync_rate: 20
        learning_rate: 0.001
        discount_factor: 0.99
        stop_on_reward: 10000
        fc1_nodes: 512
        """  
        self.env_name = config['env_name']
        self.replay_buffer_size = config['replay_buffer_size']
        self.mini_batch_size = config['mini_batch_size']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.network_sync_rate = config['network_sync_rate']
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']
        self.stop_on_reward = config['stop_on_reward']
        self.fc1_nodes = config['fc1_nodes']

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam

        self.LOG_FILE = os.path.join(RUNS_DIR_PATH, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR_PATH, f"{hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR_PATH, f"{hyperparameter_set}.png")

    #We will use this for both training and testing
    def run(self, is_training=True, render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gymnasium.make(self.env_name, render_mode="human" if render else None)


        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_DQN = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemoryBuffer(max_size=self.replay_buffer_size)
            
            epsilon = self.epsilon_start

            target_DQN = DQN(num_states, num_actions).to(device)
            target_DQN.load_state_dict(policy_DQN.state_dict())

            #We well use this to update the target network
            step_counter = 0

            self.optimizer = torch.optim.Adam(policy_DQN.parameters(), lr=self.learning_rate)

            best_reward = -9999
        else:
            policy_DQN.load_state_dict(torch.load(self.MODEL_FILE))
            policy_DQN.eval()  # Set the model to evaluation mode

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            episode_reward = 0

            ## We will train indefinitely, and close the environment manually based on rewards per episode data.
            while (not terminated and episode_reward < self.stop_on_reward):
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

            if is_training and episode_reward >= self.stop_on_reward:
                print(f"✅ Training complete: Reached reward {episode_reward} at episode {episode}")
                torch.save(policy_DQN.state_dict(), self.MODEL_FILE)
                break



            # Save the model when new best reward is achieved
            if is_training and episode_reward > best_reward:
                log_message = f"New best reward: {episode_reward} at episode {episode}"
                print(log_message)
                with open(self.LOG_FILE, 'a') as log_file:
                    log_file.write(log_message + '\n')
                torch.save(policy_DQN.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            if is_training:
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                epsilon_history.append(epsilon)

                if len(memory) > self.mini_batch_size:
                    # Sample a mini-batch from the memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(policy_DQN, target_DQN, mini_batch) ## This is where training happens

                    if step_counter > self.network_sync_rate:
                        target_DQN.load_state_dict(policy_DQN.state_dict())
                        step_counter = 0
            
            # Run only one episode in the testing mode
            if not is_training:
                print(f"Episode {episode} finished with reward: {episode_reward}")
                break

        if is_training:
            self.save_graph(rewards_per_episode, epsilon_history)

    def optimize(self, policy_DQN, target_DQN, mini_batch):

        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device) 
        rewards = torch.stack(rewards).to(device)
        new_states = torch.stack(new_states).to(device)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate the target Q-values (expected future rewards)
            target_q = rewards + (1 - terminations) * self.discount_factor * target_DQN(new_states).max(dim=1)[0]

        current_q = policy_DQN(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the agent.')
    parser.add_argument('hyperparameter', help='')
    parser.add_argument('--train', help='Run in training mode', action='store_true')
    args = parser.parse_args()

    dql  = Agent(hyperparameter_set=args.hyperparameter)
    if args.train:
        dql.run(is_training=True, render=False)  # Set render to false to train faster
    else:
        dql.run(is_training=False, render=True)  # Set render to true to see the agent play the game