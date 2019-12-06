from lib._class.agent.QNetworkAgent import QNetworkAgent

class DQNAgent(QNetworkAgent):
    '''
    Deep Q-Network Agent
    '''
    def __init__(self, env,
                 pretrain_size=1, sample_size=1, memory_size=1, model_file='DQN_MODEL.H5',
                 neurons=[1_024, 512, 256]):
        
        super().__init__(env, pretrain_size=pretrain_size, sample_size=sample_size, memory_size=memory_size,
                         model_file=model_file, neurons=neurons)