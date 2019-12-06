import lib._util.fileproc as fileproc

import numpy as np
import pandas as pd

from ast import literal_eval

class MatrixModel:
    def __init__(self, state_size, action_size, init_value=.0):
        self.state_size  = state_size
        self.action_size = action_size
        self.init_value  = init_value
        
        self.states = np.empty((0, self.state_size))
        self.values = np.empty((0, self.action_size))
        
    def state_values(self, state):
        try:
            index  = self.states.tolist().index(state.tolist())
            values = self.values[index]
        except:
            self.states = np.append(self.states, [state], axis=0)
            self.values = np.append(self.values, [[self.init_value for _ in range(self.action_size)]], axis=0)
            values      = self.state_values(state)
            
        return values
    
    def state_action_value(self, state, action):
        values = self.state_values(state)
        return values[action]
    
    def set_state_action(self, state, action, value):
        values = self.state_values(state)
        values[action] = value
        
        # No need to update array as it's modified on referencing variable
        # index = self.states.tolist().index(state.tolist())
        # self.values[index] = values
        
    def set_state(self, state, values):
        for index, value in enumerate(values):
            self.set_state_action(state, index, value)
        
    def save(self, out_path, filename):
        matrix_dict = {tuple(self.states[i]): self.values[i] for i, x in enumerate(self.states)}
        matrix_df   = pd.DataFrame.from_dict(matrix_dict, orient='index', columns=[x for x in range(self.action_size)])
        matrix_df.index.name = 'State'
        
        fileproc.generate_csv(matrix_df, out_path, filename, export_index=True)
        
    def load_model(self, source_path, filename):
        source_file = f'{source_path}{filename}'
        
        chunk_size = 50_000
        df_chunks  = pd.read_csv(source_file, sep=';', index_col=['State'],
                                 converters={'State': literal_eval},
                                 chunksize=chunk_size)
        matrix_df  = pd.concat(df_chunks)
        
        self.states = np.array([x for x in matrix_df.index.values])
        self.values = matrix_df.values