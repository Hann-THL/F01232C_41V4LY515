import numpy as np
import random

class Agent:
    def __init__(self, env):
        self.hyperparams_dict = {
            'epsilon': {
                'min': np.nan, 'max': np.nan, 'decay': np.nan, 'value': np.nan
            },
            'alpha': {
                'min': np.nan, 'max': np.nan, 'decay': np.nan, 'value': np.nan
            },
            'gamma': {
                'min': np.nan, 'max': np.nan, 'increase': np.nan, 'value': np.nan
            }
        }
        self.env = env
        
    def choose_action(self, state):
        return random.choice(self.env.available_actions())
    
    def learn(self, experience, next_action, episode):
        return False
    
    def save_model_checkpoint(self, out_path):
        pass