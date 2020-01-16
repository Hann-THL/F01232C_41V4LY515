import numpy as np
import pandas as pd

# Scikit-Learn
from sklearn.preprocessing import StandardScaler

# Reference: https://towardsdatascience.com/skewed-data-a-problem-to-your-statistical-model-9a6b5bb74e37
def transform_skewed(df, columns, drop=False):
    new_df = df.copy()
    
    for column in columns:
        new_df[f'{column}_log'] = np.log1p(df[column])
        
    if drop:
        new_df.drop(columns=columns, inplace=True)
    
    return new_df

def standard_scaler(df, columns, drop=False):
    new_df = df.copy()
    
    values = StandardScaler().fit_transform(df[columns])
    new_df = pd.concat([
        new_df,
        pd.DataFrame(values, columns=[f'ss_{x}' for x in columns])
    ], axis=1)
    
    if drop:
        new_df.drop(columns=columns, inplace=True)
    
    return new_df