from lib._class.agent.DQNAgent import DQNAgent

import numpy as np

class D2QNAgent(DQNAgent):
    '''
    Double DQN Agent
    '''
    def __init__(self, env,
                 pretrain_size=1, sample_size=1, memory_size=1, model_file='D2QN_MODEL.H5',
                 neurons=[1_024, 512, 256]):
        
        super().__init__(env, pretrain_size=pretrain_size, sample_size=sample_size, memory_size=memory_size,
                         model_file=model_file, neurons=neurons)
        
        # Target Network
        self.hyperparams_dict['tau'] = .01
        self.target_model = self.build_model(self.env.state_size(), self.env.action_size(),
                                             self.hyperparams_dict['alpha']['value'], neurons)
        self.target_model.set_weights(self.main_model.get_weights())
        
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
        
        # Fixed Q-Target
        # Reference: https://www.youtube.com/watch?v=UCgsv6tMReY
        # Get Q-values for states in [[q1,q2,q3],[q1,q2,q3]...[q1,q2,q3]] format
        q_values      = self.target_model.predict(next_states)
        next_q_values = self.main_model.predict(next_states)
        q_targets     = self.main_model.predict(states)
        
        # Calculate Q-targets
        max_q_indexes  = np.argmax(next_q_values, axis=1)
        sample_indexes = np.arange(self.sample_size, dtype=np.int32)
        q_targets[sample_indexes, action_indexes] = rewards + self.hyperparams_dict['gamma']['value'] * q_values[sample_indexes, max_q_indexes.astype(int)] * terminals
        
        # Update Q-targets
        self.main_model.train(states, q_targets)
        
        # Update target network weights
        # Reference: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
        main_weights   = self.main_model.get_weights()
        target_weights = self.target_model.get_weights()
        for weight_index in range(len(target_weights)):
            target_weights[weight_index] = main_weights[weight_index] * self.hyperparams_dict['tau'] + target_weights[weight_index] * (1 - self.hyperparams_dict['tau'])
        self.target_model.set_weights(target_weights)
        
        # Adjust hyperparameters
        if done:
            self.adjust_hyperparams('epsilon', episode)
            self.adjust_hyperparams('alpha', episode)
            self.adjust_hyperparams('gamma', episode)
            
            # Update optimizer learning rate
            if self.hyperparams_dict['alpha']['decay'] != 0:
                self.main_model.set_learning_rate(self.hyperparams_dict['alpha']['value'])
        
        return True