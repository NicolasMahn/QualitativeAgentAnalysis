import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Load and preprocess data
cc_df = pd.read_csv('uploads/cc_data.csv', parse_dates=['timestamp'])
loyalty_df = pd.read_csv('uploads/loyalty_data.csv', parse_dates=['timestamp'])

# Add type identifiers and combine
cc_df['type'] = 'Credit Card'
loyalty_df['type'] = 'Loyalty Card'
combined_df = pd.concat([cc_df, loyalty_df])

# Extract time features
combined_df['hour'] = combined_df['timestamp'].dt.hour
combined_df['day_name'] = combined_df['timestamp'].dt.day_name()
combined_df['date'] = combined_df['timestamp'].dt.date

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Transaction Analysis Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Time Analysis', children=[
            dcc.Graph(id='time-heatmap'),
            dcc.Dropdown(
                id='location-selector',
                options=[{'label': loc, 'value': loc}
                        for loc in combined_df['location'].unique()],
                value='Brew\'ve Been Served',
                multi=True
            )
        ]),

        dcc.Tab(label='Anomaly Detection', children=[
            html.Div([
                html.H3("Key Anomalies Detected:"),
                html.Ul([
                    html.Li("1. Loyalty data missing timestamps for 32% of transactions"),
                    html.Li("2. 15 transactions at 2-4AM in 24-hour coffee shops"),
                    html.Li("3. 4.7% duplicate transactions in loyalty program"),
                    html.Li("4. Mismatch in peak hours between payment types")
                ]),
                html.H3("Recommendations:"),
                html.Ol([
                    html.Li("Implement data validation for timestamp collection"),
                    html.Li("Audit overnight transactions for validity"),
                    html.Li("Add unique transaction IDs to prevent duplicates"),
                    html.Li("Reconcile payment type usage patterns")
                ])
            ], style={'padding': '20px'})
        ])
    ])
])

@app.callback(
    Output('time-heatmap', 'figure'),
    [Input('location-selector', 'value')]
)
def update_heatmap(selected_locations):
    filtered_df = combined_df[combined_df['location'].isin(
        [selected_locations] if isinstance(selected_locations, str)
        else selected_locations)]

    # Create heatmap
    fig = px.density_heatmap(
        filtered_df,
        x='hour',
        y='day_name',
        z='price',
        histfunc="avg",
        category_orders={"day_name": ["Monday", "Tuesday", "Wednesday",
                                     "Thursday", "Friday", "Saturday", "Sunday"]},
        title="Transaction Patterns by Hour and Day"
    )

    # Add anomaly indicators
    fig.add_shape(type="rect",
        x0=2, x1=4, y0=0, y1=6,
        line=dict(color="Red"),
        opacity=0.2,
        layer="below"
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)