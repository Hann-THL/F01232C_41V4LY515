from lib._class.agent.SarsaAgent import SarsaAgent

import numpy as np
import random

class SarsaLambdaAgent(SarsaAgent):
    '''
    SARSA (λ) Agent
    '''
    def __init__(self, env, model_file='SARSA_Lambda_MODEL.CSV', episodic_trace=False):
        super().__init__(env, model_file=model_file)
        
        # Eligibility Trace
        self.hyperparams_dict['elig_trace'] = {
            'init': 1 / self.env.action_size(),
            'lambda': .9
        }
        self.e_model        = self.__build_e_model()
        self.episodic_trace = episodic_trace
        
    def __build_e_model(self):
        return self.build_model(self.env.state_size(), self.env.action_size(),
                                init_value=self.hyperparams_dict['elig_trace']['init'])
        
    def choose_action(self, state):
        # Exploration
        if np.random.uniform(0, 1) <= self.hyperparams_dict['epsilon']['value']:
            return random.choice(self.env.available_actions())
        
        # Exploitation
        else:
            q_values = self.main_model.state_values(state).copy()
            # Ensure state added to main model is added to elibility trace as well
            self.e_model.state_values(state)
            return self.exploitation(q_values)
        
    def learn(self, experience, next_action, episode):
        state, action, reward, next_state, done = experience
        
        # Reference:
        # https://naifmehanna.com/2018-10-18-implementing-sarsa-in-python/
        next_q_value = self.main_model.state_action_value(next_state, next_action)
        q_value = self.main_model.state_action_value(state, action)
        
        # Ensure state added to main model is added to elibility trace as well
        self.e_model.state_action_value(next_state, next_action)
        
        # Accumulate Trace
        # e_value = self.e_model.state_action_value(state, action)
        # self.e_model.set_state_action(state, action, e_value +1)
        
        # Replace Trace
        self.e_model.set_state(state, np.zeros(self.env.action_size()))
        self.e_model.set_state_action(state, action, 1)
        
        states_q_values = self.main_model.values
        states_e_values = self.e_model.values
        
        # Calculate & Update Q-matrix
        q_target = reward + self.hyperparams_dict['gamma']['value'] * next_q_value
        td_error = q_target - q_value
        
        states_q_values = states_q_values + self.hyperparams_dict['alpha']['value'] * td_error * states_e_values
        self.main_model.values = np.round(states_q_values, 5)
        
        # Decay & Update E-matrix
        states_e_values = states_e_values * self.hyperparams_dict['gamma']['value'] * self.hyperparams_dict['elig_trace']['lambda']
        self.e_model.values = np.round(states_e_values, 5)
        
        # Adjust hyperparameters
        if done:
            self.adjust_hyperparams('epsilon', episode)
            self.adjust_hyperparams('alpha', episode)
            self.adjust_hyperparams('gamma', episode)
            
            # Re-initialize Eligibility Trace on each episode:
            # https://stackoverflow.com/questions/29904270/eligibility-trace-reinitialization-between-episodes-in-sarsa-lambda-implementati
            if self.episodic_trace:
                self.e_model = self.__build_e_model()
                self.e_model.states = self.main_model.states.copy()
                self.e_model.values = np.full(self.main_model.values.shape, self.hyperparams_dict['elig_trace']['init'])
                
        return True