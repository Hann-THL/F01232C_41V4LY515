from lib._class.agent.DQNAgent import DQNAgent
from lib._class.network.NNModel import NNModel

class DuelDQNAgent(DQNAgent):
    '''
    Dueling DQN Agent
    '''
    def __init__(self, env,
                 pretrain_size=1, sample_size=1, memory_size=1, model_file='DuelDQN_MODEL.H5',
                 neurons=[1_024, 512, 256]):
        
        super().__init__(env, pretrain_size=pretrain_size, sample_size=sample_size, memory_size=memory_size,
                         model_file=model_file, neurons=neurons)
        
    def build_model(self, state_size, action_size, alpha, neurons):
        return NNModel(state_size, action_size, alpha, neurons=neurons, network_type='val-adv')