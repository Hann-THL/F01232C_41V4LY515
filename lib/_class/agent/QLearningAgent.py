from lib._class.agent.Agent import Agent
from lib._class.agent.model.MatrixModel import MatrixModel

import numpy as np
import random

class QLearningAgent(Agent):
    def __init__(self, env, build_model=True, model_file='QLearning_MODEL.CSV'):
        super().__init__(env)
        
        self.hyperparams_dict = {
            'epsilon': {
                'min': .01, 'max': 1., 'decay': .0005, 'value': 1.
            },
            'alpha': {
                'min': .00001, 'max': .9, 'decay': .001, 'value': .9
            },
            'gamma': {
                'min': .1, 'max': .9, 'increase': .001, 'value': .1
            }
        }
        self.env        = env
        self.model_file = model_file
        
        if build_model:
            self.main_model = self.build_model(self.env.state_size(), self.env.action_size())
        
    def build_model(self, state_size, action_size, init_value=.0):
        return MatrixModel(state_size, action_size, init_value)
    
    def save_model_checkpoint(self, out_path):
        self.main_model.save(out_path, self.model_file)
        
    def load_model_checkpoint(self, source_path):
        self.main_model.load_model(source_path, self.model_file)
    
    def __random_argmax(self, q_values, random_if_empty=False):
        # Reference:
        # https://gist.github.com/stober/1943451
        indexes = np.nonzero(q_values == np.amax(q_values))[0]
        
        try:
            return np.random.choice(indexes)
        except TypeOfError as error:
            if not random_if_empty:
                raise Exception(error)
                
            return np.random.choice(indexes if indexes.size > 0 else q_values.size)
        
    def exploitation(self, q_values):
        # Validation whether action with highest Q-value is valid
        actions      = self.env.action_space()
        action_index = self.__random_argmax(q_values)
        action       = actions[action_index]
        
        if action not in self.env.available_actions():
            # Remove action if it's not a valid action on current state
            q_values = np.delete(q_values, action_index)
            del actions[action_index]

            # Select action with 2nd highest Q-value
            action_index = self.__random_argmax(q_values)
            action       = actions[action_index]
        
        return action
        
    def choose_action(self, state):
        # Exploration
        if np.random.uniform(0, 1) <= self.hyperparams_dict['epsilon']['value']:
            return random.choice(self.env.available_actions())
        
        # Exploitation
        else:
            q_values = self.main_model.state_values(state).copy()
            return self.exploitation(q_values)
        
    def adjust_hyperparams(self, param_name, episode):
        if param_name in ['epsilon', 'alpha']:
            min_value = self.hyperparams_dict[param_name]['min']
            max_value = self.hyperparams_dict[param_name]['max']
            rate      = self.hyperparams_dict[param_name]['decay']
        
        elif param_name == 'gamma':
            # Swap min. max. for incrementing
            min_value = self.hyperparams_dict[param_name]['max']
            max_value = self.hyperparams_dict[param_name]['min']
            rate      = self.hyperparams_dict[param_name]['increase']
            
        self.hyperparams_dict[param_name]['value'] = min_value + (max_value - min_value) * np.exp(-rate * episode)
        
    def learn(self, experience, next_action, episode):
        state, action, reward, next_state, done = experience
        
        # Q[s, a] = Q[s, a] + alpha * (reward + gamma * Max[Q(s’, A)] - Q[s, a])
        max_q_value = np.max(self.main_model.state_values(next_state))
        q_value = self.main_model.state_action_value(state, action)
        q_value = q_value + self.hyperparams_dict['alpha']['value'] * (reward + self.hyperparams_dict['gamma']['value'] * max_q_value - q_value)
        self.main_model.set_state_action(state, action, round(q_value, 10))
        
        # Adjust hyperparameters
        if done:
            self.adjust_hyperparams('epsilon', episode)
            self.adjust_hyperparams('alpha', episode)
            self.adjust_hyperparams('gamma', episode)
            
        return True