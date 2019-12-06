from lib._class.agent.QLearningAgent import QLearningAgent

class SarsaAgent(QLearningAgent):
    def __init__(self, env, model_file='SARSA_MODEL.CSV'):
        super().__init__(env, model_file=model_file)
        
    def learn(self, experience, next_action, episode):
        state, action, reward, next_state, done = experience
        
        # Q[s, a] = Q[s, a] + alpha * (reward + gamma * Q(s’, a’) - Q[s, a])
        next_q_value = self.main_model.state_action_value(next_state, next_action)
        q_value = self.main_model.state_action_value(state, action)
        q_value = q_value + self.hyperparams_dict['alpha']['value'] * (reward + self.hyperparams_dict['gamma']['value'] * next_q_value - q_value)
        self.main_model.set_state_action(state, action, round(q_value, 10))
        
        # Adjust hyperparameters
        if done:
            self.adjust_hyperparams('epsilon', episode)
            self.adjust_hyperparams('alpha', episode)
            self.adjust_hyperparams('gamma', episode)
            
        return True