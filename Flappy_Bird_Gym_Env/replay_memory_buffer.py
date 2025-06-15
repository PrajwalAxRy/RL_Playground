from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def append(self, transition):
        """Add a new transition to the memory."""
        self.memory.append(transition)
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions from the memory."""
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)