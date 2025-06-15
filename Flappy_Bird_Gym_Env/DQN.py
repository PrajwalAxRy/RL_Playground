import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# generate main to test the DQN model
if __name__ == "__main__":
    # Example usage
    state_dim = 12  # Example state dimension
    action_dim = 2  # Example action dimension
    model = DQN(state_dim, action_dim)
    
    # Print model architecture
    print(model)
    
    # Example forward pass with random input
    state_input = torch.randn(1, state_dim) # (batch_size, dim)
    output = model(state_input)
    print("Output Q-values:", output)