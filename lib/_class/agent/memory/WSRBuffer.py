from lib._class.agent.memory.ReplayBuffer import ReplayBuffer

import numpy as np

class WSRBuffer(ReplayBuffer):
    '''
    Weighted Sample Replay
    '''
    def __init__(self, memory_size, state_size, action_size, unusual_sample_factor=.99):
        super().__init__(memory_size, state_size, action_size)
        
        # Reference: https://medium.com/ml-everything/reinforcement-learning-with-sparse-rewards-8f15b71d18bf
        # Determine how much difference between experiences with low rewards and experiences with high rewards
        # The lower the value, the higher then difference, and 1 means no different
        self.unusual_sample_factor = unusual_sample_factor
        
    def sample(self, batch_size, experience=None):
        # Sort rewards index descendingly by absolute value
        sort_indexes = np.argsort(-np.abs(self.rewards[:self.sample_counter]))
        
        # Calculate probabilities and ensure sum of probabilities is max. 1
        probabilities = np.power(self.unusual_sample_factor, np.arange(len(sort_indexes)))
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample minibatch
        batch_size = batch_size if experience is None else batch_size -1
        minibatch  = np.random.choice(np.arange(len(probabilities)), size=batch_size, p=probabilities)
        minibatch_indexes = sort_indexes[minibatch]
        
        states      = self.states[minibatch_indexes]
        actions     = self.actions[minibatch_indexes]
        rewards     = self.rewards[minibatch_indexes]
        next_states = self.next_states[minibatch_indexes]
        terminals   = self.terminals[minibatch_indexes]
        
        return self.currexp_to_sample(experience, (states, actions, rewards, next_states, terminals))