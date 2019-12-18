from lib._class.agent.QLearningAgent import QLearningAgent
from lib._class.agent.memory.ReplayBuffer import ReplayBuffer
from lib._class.agent.memory.WSRBuffer import WSRBuffer
from lib._class.network.NNModel import NNModel

import numpy as np
import random

class QNetworkAgent(QLearningAgent):
    def __init__(self, env, build_model=True,
                 pretrain_size=None, sample_size=1, memory_size=1, model_file='QNetwork_MODEL.H5',
                 neurons=[]):
        
        super().__init__(env, build_model=False, model_file=model_file)
        
        self.hyperparams_dict = {
            'epsilon': {
                'min': .1, 'max': 1., 'decay': .00025, 'value': 1.
            },
            'alpha': {
                'min': .00001, 'max': .00001, 'decay': .0, 'value': .00001
            },
            'gamma': {
                'min': .9, 'max': .9, 'increase': .0, 'value': .9
            }
        }
        self.memory        = ReplayBuffer(memory_size, self.env.state_size(), self.env.action_size())
        # self.memory        = WSRBuffer(memory_size, self.env.state_size(), self.env.action_size(), unusual_sample_factor=.99)
        self.sample_size   = sample_size
        self.pretrain_size = pretrain_size if pretrain_size is not None else self.sample_size
        
        if build_model:
            self.main_model = self.build_model(self.env.state_size(), self.env.action_size(),
                                               self.hyperparams_dict['alpha']['value'], neurons)
        
    def build_model(self, state_size, action_size, alpha, neurons):
        return NNModel(state_size, action_size, alpha, neurons=neurons)
    
    def save_model_checkpoint(self, out_path):
        self.main_model.save_model_checkpoint(out_path, self.model_file)
        
    def load_model_checkpoint(self, source_path):
        self.main_model.load_model_checkpoint(source_path, self.model_file)
    
    def choose_action(self, state):
        # Exploration
        if np.random.uniform(0, 1) <= self.hyperparams_dict['epsilon']['value']:
            return random.choice(self.env.available_actions())
        
        # Exploitation
        else:
            # Change [obs1,obs2,obs3] to [[obs1,obs2,obs3]] format
            state = state[np.newaxis, :]
            
            # Get Q-values for current state in [[q1,q2,q3]] format
            q_values = self.main_model.predict(state)
            
            # Change [[q1,q2,q3]] to [q1,q2,q3] format
            q_values = q_values.reshape(-1)
            
            return self.exploitation(q_values)
        
    def learn(self, experience, next_action, episode):
        state, action, reward, next_state, done = experience
        self.memory.store_transition(state, action, reward, next_state, done)
        
        if self.memory.counter < self.pretrain_size:
            return False
        
        # states:      [[obs1,obs2,obs3],[obs1,obs2,obs3]...[obs1,obs2,obs3]]
        # actions:     [[0,0,1],[0,1,0]...[1,0,0]]
        # rewards:     [r1,r2...rN]
        # next_states: [[obs1,obs2,obs3],[obs1,obs2,obs3]...[obs1,obs2,obs3]]
        # terminals:   [t1,t2...tN]
        states, actions, rewards, next_states, terminals = self.memory.sample(self.sample_size, experience=experience)
        
        # Change actions from [[0,0,1],[0,1,0]...[1,0,0]] to [2,1...0] format
        action_values  = np.array(self.env.action_space(), dtype=np.int8)
        action_indexes = np.dot(actions, action_values)
        
        # Reference: https://www.youtube.com/watch?v=5fHngyN8Qhw
        # Get Q-values for states in [[q1,q2,q3],[q1,q2,q3]...[q1,q2,q3]] format
        q_values      = self.main_model.predict(states)
        next_q_values = self.main_model.predict(next_states)
        q_targets     = q_values.copy()
        
        # Calculate Q-targets
        sample_indexes = np.arange(self.sample_size, dtype=np.int32)
        q_targets[sample_indexes, action_indexes] = rewards + self.hyperparams_dict['gamma']['value'] * np.max(next_q_values, axis=1) * terminals
        
        # Update Q-targets
        self.main_model.train(states, q_targets)
        
        # Adjust hyperparameters
        if done:
            self.adjust_hyperparams('epsilon', episode)
            self.adjust_hyperparams('alpha', episode)
            self.adjust_hyperparams('gamma', episode)
            
            # Update optimizer learning rate
            if self.hyperparams_dict['alpha']['decay'] != 0:
                self.main_model.set_learning_rate(self.hyperparams_dict['alpha']['value'])
        
        return True