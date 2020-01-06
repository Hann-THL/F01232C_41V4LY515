import lib._util.fileproc as fp

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

import numpy as np

def fast_stat(df):
    na_count   = df.isna().sum()
    na_percent = na_count / len(df) * 100
    
    stats = [f'{x[0][0] :<30} | {str(x[0][1]) :<15} | {x[1][1] :<10,.5f} | {x[2][1] :,}'
             for x in zip(df.dtypes.items(), na_percent.items(), na_count.items())]
    
    print(df.shape)
    print('Column                         | Type            | N/A %      | N/A Count')
    print('-------------------------------------------------------------------------')
    print('\n'.join(stats))

def figure(data, title=None, xlabel=None, ylabel=None):
    axis_dict = dict(
        title=xlabel,
        gridcolor='rgb(159, 197, 232)'
    )
    xaxis_kwargs = {f'xaxis{x+1 if x != 0 else ""}': axis_dict
                    for x in range(50)}
    yaxis_kwargs = {f'yaxis{x+1 if x != 0 else ""}': axis_dict
                    for x in range(50)}

    layout = go.Layout(
    	**xaxis_kwargs,
    	**yaxis_kwargs,
        title = title,
        hovermode='x',
        showlegend=True,
        legend_orientation='h',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )

    return go.Figure(data=data, layout=layout)

def update_axis(fig, axis_count, gridcolor='rgb(159, 197, 232)'):
    for x in range(axis_count):
        suffix = x+1 if x != 0 else ''
        fig['layout'][f'xaxis{suffix}']['gridcolor'] = gridcolor
        fig['layout'][f'yaxis{suffix}']['gridcolor'] = gridcolor

def plot_graph(data, title, xlabel=None, ylabel=None, generate_file=True, out_path=None):
    fig = figure(data, title, xlabel, ylabel)

    if generate_file:
        generate_plot(fig, out_path, title)
    else:
        generate_plot(fig)

def generate_plot(fig, out_path=None, out_filename=None, axis_count=None):
    if axis_count is not None:
        update_axis(fig, axis_count)

    if out_path is None or out_filename is None:
        iplot(fig)
    else:
        fp.create_directory(out_path)
        out_filename = f'{out_filename}.html' if '.html' not in out_filename else out_filename
        out_file     = f'{out_path}{out_filename}'
        plot(fig, filename=out_file, auto_open=False)
        
        print(f'Generated: {out_file}')

def plot_subplots(data, max_col, title, subplot_titles=None, out_path=None, showlegend=False, layout_height=None):
    max_row = int(np.ceil(len(data) / max_col))
    
    fig = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles, x_title=title)

    for index, trace in enumerate(data):
        col = index +1

        if col <= max_col:
            row = 1
        else:
            quotient = int(col / max_col)
            col -= (max_col * quotient)
            if col == 0:
                col = max_col
            elif col == 1:
                row += 1

        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(showlegend=showlegend, height=layout_height, plot_bgcolor='rgba(0, 0, 0, 0)')
    generate_plot(fig, out_path=out_path, out_filename=title, axis_count=len(data))

def datagroups_subplots(data_groups, max_col, title, subplot_titles=None, out_path=None, showlegend=False, layout_height=None):
    max_row = int(np.ceil(len(data_groups) / max_col))
    
    fig = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles, x_title=title)
    for index, data_group in enumerate(data_groups):
        col = index +1

        if col <= max_col:
            row = 1
        else:
            quotient = int(col / max_col)
            col -= (max_col * quotient)
            if col == 0:
                col = max_col
            elif col == 1:
                row += 1

        for data in data_group:
            fig.add_trace(data, row=row, col=col)

    fig.update_layout(showlegend=showlegend, height=layout_height, plot_bgcolor='rgba(0, 0, 0, 0)', barmode='overlay')
    generate_plot(fig, out_path=out_path, out_filename=title, axis_count=len(data_groups))

def histogram(df, title=None, out_path=None, layout_height=None):
    data = []

    for column in df.columns:
        data.append(go.Histogram(
            x = df[column]
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in df.columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_height=layout_height)

def boxplot(df, title, out_path=None, layout_height=None, numeric_only=True):
    data = []

    columns = [k for k,v in df.dtypes.items() if 'float' in str(v) or 'int' in str(v)] if numeric_only else df.columns
    for column in columns:
        data.append(go.Box(
            y = df[column],
            boxpoints='outliers', # all, outliers, suspectedoutliers
            boxmean=True
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_height=layout_height)

def violinplot(df, title, out_path=None, layout_height=None, numeric_only=True):
    data = []

    columns = [k for k,v in df.dtypes.items() if 'float' in str(v) or 'int' in str(v)] if numeric_only else df.columns
    for column in columns:
        data.append(go.Violin(
            y=df[column],
            box_visible=True,
            meanline_visible=True,
            points='outliers' # all, outliers, suspectedoutliers
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_height=layout_height)