import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
import math

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive 3D Function Visualizer", style={'textAlign': 'center'}),

    html.Div([
        # Function input
        dcc.Input(
            id='function-input',
            type='text',
            placeholder='Enter function (e.g., z = 9 - x**2 - y**2)',
            style={'width': '300px', 'margin': '5px'}
        ),
        # Area conditions input
        dcc.Input(
            id='area-conditions',
            type='text',
            placeholder='Enter area conditions (e.g., y <= 1 - x, x >= 0, y >= 0)',
            style={'width': '400px', 'margin': '5px'}
        ),
        # Range inputs
        html.Div([
            dcc.Input(id='x-min', type='number', placeholder='X min (0)', style={'width': '100px', 'margin': '5px'}),
            dcc.Input(id='x-max', type='number', placeholder='X max (1)', style={'width': '100px', 'margin': '5px'}),
            dcc.Input(id='y-min', type='number', placeholder='Y min (0)', style={'width': '100px', 'margin': '5px'}),
            dcc.Input(id='y-max', type='number', placeholder='Y max (1)', style={'width': '100px', 'margin': '5px'}),
        ]),
        # Control button
        html.Button('Update Plot', id='update-button', n_clicks=0),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='3d-plot', style={'height': '80vh'}),
    html.Div(id='volume-display', style={'textAlign': 'center', 'fontSize': '20px'})
])


def safe_eval(expr, X, Y):
    """Safely evaluate mathematical expressions"""
    allowed = {'x': X, 'y': Y, 'math': math, 'np': np}
    try:
        return eval(expr, {'__builtins__': None}, allowed)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return np.full_like(X, True)


def parse_conditions(conditions, X, Y):
    """Parse multiple area conditions"""
    if not conditions:
        return np.full_like(X, True)

    mask = np.full_like(X, True)
    for condition in conditions.split(','):
        condition = condition.strip().replace('^', '**')
        if not condition:
            continue
        try:
            mask &= safe_eval(condition, X, Y)
        except Exception as e:
            print(f"Condition error: {e}")
    return mask


@app.callback(
    [Output('3d-plot', 'figure'),
     Output('volume-display', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('function-input', 'value'),
     State('area-conditions', 'value'),
     State('x-min', 'value'),
     State('x-max', 'value'),
     State('y-min', 'value'),
     State('y-max', 'value')]
)
def update_plot(n_clicks, func_str, conditions, x_min, x_max, y_min, y_max):
    # Handle default values
    x_min = x_min if x_min is not None else 0
    x_max = x_max if x_max is not None else 1
    y_min = y_min if y_min is not None else 0
    y_max = y_max if y_max is not None else 1

    # Generate grid
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)

    # Parse function
    try:
        func_str = func_str or 'x**2 + y**2'
        func_str = func_str.replace('^', '**').replace('z=', '').strip()
        Z = eval(func_str, {'np': np, 'math': math, 'x': X, 'y': Y})
    except Exception as e:
        print(f"Function error: {e}")
        Z = X ** 2 + Y ** 2

    # Parse area conditions
    area_mask = parse_conditions(conditions, X, Y)
    Z_masked = np.where(area_mask, Z, np.nan)

    # Calculate volume
    dx = (x_max - x_min) / 49
    dy = (y_max - y_min) / 49
    volume = np.nansum(Z_masked * dx * dy)

    # Create figure
    fig = go.Figure()

    # Add surface plot
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_masked,
        colorscale='viridis',
        opacity=0.8,
        showscale=True,
        name='Function Surface'
    ))

    # Add shaded area
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale=[[0, 'rgba(255,0,0,0.3)']],
        showscale=False,
        surfacecolor=area_mask.astype(int),
        name='Integration Area'
    ))

    # Update layout
    fig.update_layout(
        title=f"Visualizing: {func_str}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.6))
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig, f"Approximate Volume: {volume:.4f}"


if __name__ == '__main__':
    app.run(debug=True, port=8051, use_reloader=False)