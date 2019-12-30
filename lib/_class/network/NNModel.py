import lib._util.fileproc as fileproc
from lib._class.network.generator.DataGenerator import DataGenerator

import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.constraints import max_norm

class NNModel:
    def __init__(self, state_size, action_size, alpha, neurons=[], network_type='default'):
        self.state_size   = state_size
        self.action_size  = action_size
        self.neurons      = neurons
        self.network_type = network_type
        
        # Input layer
        inputs = self.__input_layer()
        
        # Hidden layer
        layer = self.__hidden_layer(inputs)
        
        # Output layer
        outputs = self.__output_layer(layer)
    
        self.model = Model(inputs=inputs, outputs=outputs)
        # Reference:
        # - https://stats.stackexchange.com/questions/351409/difference-between-rho-and-decay-arguments-in-keras-rmsprop
        # - https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping
        # - https://stackoverflow.com/questions/42264567/keras-ml-library-how-to-do-weight-clipping-after-gradient-updates-tensorflow-b
        self.model.compile(optimizer=RMSprop(lr=alpha, clipnorm=5., rho=.95, epsilon=.01), loss=MeanSquaredError())
        # self.model.compile(optimizer=Adam(lr=alpha, clipnorm=5., rho=.95, epsilon=.01), loss=MeanSquaredError())
        
        # Reference: https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4
        # self.model.compile(optimizer=RMSprop(lr=alpha, clipnorm=5., rho=.95, epsilon=.01), loss=Huber(delta=1.0))
        # self.model.compile(optimizer=Adam(lr=alpha, clipnorm=5., rho=.95, epsilon=.01), loss=Huber(delta=1.0))
    
    def __input_layer(self):
        if self.network_type == 'conv1d':
            # Reference: https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57
            # Convolutional layer
            inputs = Input(shape=(None, self.state_size), name='Input')
            inputs = Conv1D(filters=32, kernel_size=2, padding="same", name='Conv1D')(inputs)
            inputs = Activation('relu', name='ReLU')(inputs)
            inputs = GlobalMaxPooling1D(name='MaxPooling')(inputs)
            inputs = Conv1D(filters=64, kernel_size=2, padding="same", name='Conv1D')(inputs)
            inputs = Activation('relu', name='ReLU')(inputs)
            inputs = GlobalMaxPooling1D(name='MaxPooling')(inputs)
            inputs = Flatten(name='Flatten')(inputs)
            
        else:
            inputs = Input(shape=(self.state_size,), name='Input')
            
        return inputs
    
    def __hidden_layer(self, connected_layer):
        if len(self.neurons) == 0:
            return connected_layer
        
        for index, neuron in enumerate(self.neurons):
            # Reference:
            # - https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/
            # - https://vincentblog.xyz/posts/dropout-and-batch-normalization
            layer = Dense(neuron,
                          kernel_initializer='he_uniform',
                          kernel_constraint=max_norm(5),
                          use_bias=False,
                          name=f'Hidden_{index}')(connected_layer if index == 0 else layer)
            
            # Reference: https://medium.com/luminovo/a-refresher-on-batch-re-normalization-5e0a1e902960
            layer = BatchNormalization(scale=False, renorm=True, renorm_clipping={ 'rmax': 1, 'rmin': 0, 'dmax': 0 })(layer)
            
            # Reference: https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7
            layer = Activation('relu')(layer)
            # layer = LeakyReLU(alpha=.001)(layer)
            # layer = ELU()(layer)
            # layer = ReLU(max_value=6)(layer)

            layer = Dropout(rate=.2)(layer)
        return layer
    
    def __output_layer(self, connected_layer):
        if self.network_type == 'val-adv':
            # Reference: https://www.reddit.com/r/reinforcementlearning/comments/bu02ej/help_with_dueling_dqn/
            # Value & Advantage Layer
            val = Dense(self.action_size, name='Value')(connected_layer)
            val = Activation('linear')(val)
            adv = Dense(self.action_size, name='Advantage')(connected_layer)
            adv = Activation('linear')(adv)

            # Output layer
            mean    = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name='Mean')(adv)
            adv     = Subtract(name='Advantage_Mean')([adv, mean])
            outputs = Add(name='Value_Advantage')([val, adv])
        
        else:
            outputs = Dense(self.action_size,
                            kernel_initializer='he_uniform',
                            name='Output')(connected_layer)
            # outputs = BatchNormalization(scale=False, renorm=True, renorm_clipping={ 'rmax': 1, 'rmin': 0, 'dmax': 0 })(outputs)
            outputs = Activation('linear')(outputs)
            
        return outputs
    
    def model_diagram(self, out_path, filename):
        fileproc.create_directory(out_path)
        plot_model(self.model, to_file=f'{out_path}{filename}.png', show_shapes=True, rankdir='LR')
        
    def save_model_checkpoint(self, out_path, filename):
        fileproc.create_directory(out_path)
        self.model.save(f'{out_path}{filename}')
        
    def load_model_checkpoint(self, source_path, filename):
        self.model = load_model(f'{source_path}{filename}')
        
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
        
    def set_learning_rate(self, lr):
        K.eval(self.model.optimizer.lr.assign(lr))
        
    def train(self, inputs, targets):
        # TODO - implement early stopping
        # Reference: https://lambdalabs.com/blog/tensorflow-2-0-tutorial-04-early-stopping/
        # epochs  = 20
        epochs  = 1
        history = self.model.fit(inputs, targets, epochs=epochs, verbose=0, batch_size=len(inputs))
        
        # generator = DataGenerator(inputs, targets, batch_size=1)
        # history   = self.model.fit_generator(generator, epochs=epochs, verbose=0,
        #                                      workers=4, steps_per_epoch=1, max_queue_size=1)
        return history
    
    def predict(self, inputs):
        return self.model.predict(inputs, batch_size=len(inputs))