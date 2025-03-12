import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go
import math

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive 3D Function Visualizer", style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(
            id='function-input',
            type='text',
            placeholder='Enter function (e.g., z = 9 - x^2 - y^2)',
            style={'width': '400px', 'marginRight': '10px'}
        ),
        html.Button('Toggle Area Visibility', id='toggle-button', n_clicks=0)
    ], style={'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='3d-plot', style={'height': '80vh'})
])


def safe_eval_function(func_str):
    """Safely evaluate user-input function with limited namespace."""
    allowed_names = {'x': None, 'y': None, 'np': np, 'math': math}
    try:
        # Clean input and convert to lambda function
        func_str = func_str.lower().replace('z=', '').strip().replace('^', '**')
        code = f'lambda x, y: {func_str}'

        # Restrict builtins and use allowed names
        restricted_globals = {'__builtins__': None, 'np': np, 'math': math}
        return eval(code, restricted_globals, allowed_names)
    except:
        return lambda x, y: x ** 2 + y ** 2  # Default function


@app.callback(
    Output('3d-plot', 'figure'),
    [Input('function-input', 'value'),
     Input('toggle-button', 'n_clicks')]
)
def update_graph(func_input, n_clicks):
    # Get function from input or use default
    func = safe_eval_function(func_input) if func_input else lambda x, y: x ** 2 + y ** 2

    # Generate grid data
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)

    try:
        Z = func(X, Y)
    except:
        Z = X ** 2 + Y ** 2  # Fallback if evaluation fails

    # Mask area outside triangle
    mask = (X + Y) > 1
    Z_masked = np.where(mask, np.nan, Z)

    # Create figure
    fig = go.Figure()

    # Add surface with toggleable opacity
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_masked,
        colorscale='viridis',
        opacity=0.8 if (n_clicks % 2 == 0) else 0.3,
        showscale=True,
        name='Surface'
    ))

    # Add triangular base
    fig.add_trace(go.Scatter3d(
        x=[0, 1, 0, 0], y=[0, 0, 1, 0], z=[0, 0, 0, 0],
        mode='lines+markers',
        line=dict(color='red', width=8),
        marker=dict(size=4),
        name='Base Triangle'
    ))

    # Update layout
    fig.update_layout(
        title=f'Function: {func_input or "z = x² + y²"}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.6))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8051)