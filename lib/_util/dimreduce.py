import lib._util.visualplot as vp

# Scikit-Learn
from sklearn.decomposition import PCA

# Prince
from prince import MCA, FAMD

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

# Reference: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
def dropcorr(df, corr_ratio=.95):
    new_df  = df.copy()
    corrmat = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper_df = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than <corr_ratio>
    columns  = [x for x in upper_df.columns if any(upper_df[x] > corr_ratio)]

    return new_df.drop(new_df[columns], axis=1)

def pca_reduction(df, columns, n_component, drop=False):
    values = df[columns]
    pca    = PCA(n_components=n_component)
    
    new_df = pd.concat([
        df,
        pd.DataFrame(pca.fit_transform(values), columns=[f'pca_{x}' for x in range(1, n_component +1)])
    ], axis=1)

    if drop:
        new_df.drop(columns=columns, inplace=True)
    
    return new_df, pca.explained_variance_

def mca_reduction(df, columns, n_component, drop=False):
    values = df[columns]
    mca    = MCA(n_components=n_component)
    
    new_df = pd.concat([
        df,
        pd.DataFrame(mca.fit_transform(values).values, columns=[f'mca_{x}' for x in range(1, n_component +1)])
    ], axis=1)
    
    if drop:
        new_df.drop(columns=columns, inplace=True)

    return new_df, mca.explained_inertia_

def famd_reduction(df, columns, n_component, drop=False):
    values = df[columns]
    famd   = FAMD(n_components=n_component)
    
    new_df = pd.concat([
        df,
        pd.DataFrame(famd.fit_transform(values).values, columns=[f'famd_{x}' for x in range(1, n_component +1)])
    ], axis=1)
    
    if drop:
        new_df.drop(columns=columns, inplace=True)

    return new_df, famd.explained_inertia_

# Reference: https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/2896/pca-for-dimensionality-reduction-not-visualization/0/free-videos
def expvar_evaluation(explained_variances, title='Explained Variance Evaluation', out_path=None):
    # Evaluate by the variance, and try to preserve variance as high as 90%
    expvar_percentages    = explained_variances / np.sum(explained_variances)
    cumexpvar_percentages = np.cumsum(expvar_percentages)

    data = []
    # Scree plot
    data.append(go.Scattergl(
        x = [x for x in range(1, len(expvar_percentages) +1)],
        y = expvar_percentages,
        mode = 'lines+markers'
    ))
    
    # Cumulative explained variance %
    data.append(go.Scattergl(
        x = [x for x in range(1, len(cumexpvar_percentages) +1)],
        y = cumexpvar_percentages,
        mode = 'lines+markers'
    ))

    subplot_titles = ['Scree Plot', 'Cumulative Explained Variance (%)']
    vp.plot_subplots(data, max_col=2, title=title, subplot_titles=subplot_titles, out_path=out_path)