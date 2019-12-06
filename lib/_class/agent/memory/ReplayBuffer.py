import numpy as np

class ReplayBuffer:
    def __init__(self, memory_size, state_size, action_size):
        self.memory_size    = memory_size
        self.counter        = 0
        self.sample_counter = 0
        
        # NOTE: specify memory size to prevent using np.append, as it is much slower while adding experience
        self.states      = np.empty((self.memory_size, state_size))
        self.actions     = np.empty((self.memory_size, action_size), dtype=np.int8)
        self.rewards     = np.empty(self.memory_size)
        self.next_states = np.empty((self.memory_size, state_size))
        self.terminals   = np.empty(self.memory_size, dtype=np.float32)
        
    def preprocess(self, experience):
        state, action, reward, next_state, done = experience
        
        # One-Hot encoding
        one_hot_action = np.zeros(self.actions.shape[1], dtype=np.int8)
        one_hot_action[action] = 1
        
        return (state, one_hot_action, reward, next_state, (1 - done))
    
    # NOTE: not to use collections.deque, as it is much slower while perform sampling
    def __deque(self, store_index, element, elements):
        try:
            elements[store_index] = element
        except IndexError:
            elements = np.append(elements, [element], axis=0)
            
        return elements
    
    def store_transition(self, state, action, reward, next_state, done):
        state, action, reward, next_state, terminal = self.preprocess((state, action, reward, next_state, done))
        
        store_index      = self.counter % self.memory_size
        self.states      = self.__deque(store_index, state, self.states)
        self.actions     = self.__deque(store_index, action, self.actions)
        self.rewards     = self.__deque(store_index, reward, self.rewards)
        self.next_states = self.__deque(store_index, next_state, self.next_states)
        self.terminals   = self.__deque(store_index, terminal, self.terminals)
        
        self.counter += 1
        if self.sample_counter < self.memory_size:
            self.sample_counter += 1
        
    def currexp_to_sample(self, experience, experiences):
        states, actions, rewards, next_states, terminals = experiences
        
        # Include current experience
        if experience is not None:
            state, action, reward, next_state, terminal = self.preprocess(experience)
            
            states      = np.vstack([states, state])
            actions     = np.vstack([actions, action])
            rewards     = np.append(rewards, reward)
            next_states = np.vstack([next_states, next_state])
            terminals   = np.append(terminals, terminal)
            
        return (states, actions, rewards, next_states, terminals)
        
    def sample(self, batch_size, experience=None):
        batch_size  = batch_size if experience is None else batch_size -1
        minibatch   = np.random.choice(self.sample_counter, batch_size, replace=False)
        
        states      = self.states[minibatch]
        actions     = self.actions[minibatch]
        rewards     = self.rewards[minibatch]
        next_states = self.next_states[minibatch]
        terminals   = self.terminals[minibatch]
        
        return self.currexp_to_sample(experience, (states, actions, rewards, next_states, terminals))