import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import plotly.graph_objects as go
import math

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("3D Function Visualizer with Math Constants", style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(
            id='function-input',
            type='text',
            placeholder='z = pi*(x² + y²)  # Try e**(-x²-y²)',
            style={'width': '300px', 'margin': '5px'}
        ),
        dcc.Input(
            id='boundary-lines',
            type='text',
            placeholder='x=0, y=0, y=1-x  # Use pi/2 for π/2',
            style={'width': '300px', 'margin': '5px'}
        ),
        html.Div([
            dcc.Input(id='x-min', type='number', placeholder='X min (0)', style={'width': '100px'}),
            dcc.Input(id='x-max', type='number', placeholder='X max (1)', style={'width': '100px'}),
            dcc.Input(id='y-min', type='number', placeholder='Y min (0)', style={'width': '100px'}),
            dcc.Input(id='y-max', type='number', placeholder='Y max (1)', style={'width': '100px'}),
        ], style={'margin': '10px'}),
        html.Button('Update Plot', id='update-button', n_clicks=0),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='3d-plot', style={'height': '75vh'}),
    html.Div(id='volume-display', style={'textAlign': 'center', 'fontSize': '20px'})
])


def parse_boundaries(boundary_str, X, Y):
    """Convert boundary lines to inequalities with math constants"""
    mask = np.full_like(X, True, dtype=bool)
    test_point = (0.25, 0.25)

    safe_dict = {
        'x': X,
        'y': Y,
        'e': math.e,
        'pi': math.pi,
        'math': math,
        'np': np
    }

    if not boundary_str:
        boundary_str = "x=0, y=0, y=1-x"

    for line in boundary_str.split(','):
        line = line.strip().replace('^', '**').replace('π', 'pi').lower()
        if '=' not in line:
            continue

        try:
            lhs, rhs = line.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Evaluate at test point
            test_value = eval(rhs, {'x': test_point[0], 'y': test_point[1], **safe_dict})

            if lhs == 'x':
                actual_value = test_point[0]
                inequality = '>=' if actual_value >= test_value else '<='
            elif lhs == 'y':
                actual_value = test_point[1]
                inequality = '>=' if actual_value >= test_value else '<='
            else:
                continue

            # Build condition with math constants
            condition = f"{lhs} {inequality} {rhs}"
            mask &= eval(condition, safe_dict)

        except Exception as e:
            print(f"Boundary error: {e}")

    return mask


@app.callback(
    [Output('3d-plot', 'figure'),
     Output('volume-display', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('function-input', 'value'),
     State('boundary-lines', 'value'),
     State('x-min', 'value'),
     State('x-max', 'value'),
     State('y-min', 'value'),
     State('y-max', 'value')]
)
def update_plot(n_clicks, func_str, boundaries, x_min, x_max, y_min, y_max):
    # Set defaults
    x_min = x_min or 0
    x_max = x_max or 1
    y_min = y_min or 0
    y_max = y_max or 1

    # Generate grid
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)

    # Create safe evaluation environment
    safe_dict = {
        'x': X,
        'y': Y,
        'e': math.e,
        'pi': math.pi,
        'math': math,
        'np': np
    }

    # Parse function with math constants
    try:
        func_str = func_str or 'x**2 + y**2'
        func_str = (func_str.replace('z=', '')
                    .replace('^', '**')
                    .replace('π', 'pi')
                    .strip())
        Z = eval(func_str, {'__builtins__': None}, safe_dict)
    except Exception as e:
        print(f"Function error: {e}")
        Z = X ** 2 + Y ** 2

    # Create area mask
    area_mask = parse_boundaries(boundaries, X, Y)
    Z_masked = np.where(area_mask, Z, np.nan)

    # Calculate volume
    dx = (x_max - x_min) / 49
    dy = (y_max - y_min) / 49
    volume = np.nansum(Z_masked * dx * dy)

    # Create plot
    fig = go.Figure()

    # Surface plot with math constants support
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_masked,
        colorscale='viridis',
        opacity=0.8,
        showscale=True,
        contours_z=dict(show=True, usecolormap=True)
    ))

    # Enhanced shaded area visualization
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale=[[0, 'rgba(255,0,0,0.3)']],
        showscale=False,
        surfacecolor=area_mask.astype(int),
        name='Integration Area'
    ))

    fig.update_layout(
        title=f"Function: {func_str}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.6))
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig, f"Volume: {volume:.4f} (Using e={math.e:.3f}, π={math.pi:.3f})"


if __name__ == '__main__':
    app.run(debug=True, port=8051, use_reloader=False)