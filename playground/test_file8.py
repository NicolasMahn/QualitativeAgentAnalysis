import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output

# Load original dataset
df = pd.read_csv("uploads/unclean_smartwatch_health_data.csv")

# 1. Analyze Data Quality Issues
def analyze_data_quality(df):
    results = {}

    # Duplicate rows
    duplicates = df.duplicated().sum()

    # Per-column analysis
    for col in df.columns:
        col_results = {}

        # Missing values
        col_results['missing'] = df[col].isna().sum()

        # Type errors
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col], errors='raise')
                col_results['type_errors'] = 0
            except:
                col_results['type_errors'] = sum(df[col].apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x)))

        # Categorical inconsistencies (for object columns)
        if df[col].dtype == 'object':
            valid_categories = {
                'Activity Level': ['Highly Active', 'Active', 'Sedentary'],
                'Stress Level': ['1', '2', '3', '4', '5']
            }.get(col, [])
            if valid_categories:
                col_results['non_standard'] = sum(~df[col].isin(valid_categories) & df[col].notna())

        # Outliers detection (for numeric columns)
        if np.issubdtype(df[col].dtype, np.number):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            col_results['outliers'] = outliers.count()

        results[col] = col_results

    return results, duplicates

# Get quality metrics
quality_metrics, total_duplicates = analyze_data_quality(df)

# Convert to DataFrame for visualization
metrics_df = pd.DataFrame.from_dict(quality_metrics, orient='index').reset_index()
metrics_df = metrics_df.rename(columns={'index': 'Column'}).melt(id_vars='Column',
                                                                var_name='Issue Type',
                                                                value_name='Count').dropna()

# 2. Create Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Smartwatch Data Quality Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Total Duplicate Rows"),
            html.P(f"{total_duplicates}", style={'fontSize': 24, 'color': 'red'})
        ], style={'padding': 20, 'border': '1px solid #ddd', 'borderRadius': 5, 'margin': 10}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Missing Values', children=[
            dcc.Graph(id='missing-values-chart')
        ]),
        dcc.Tab(label='Outliers', children=[
            dcc.Graph(id='outliers-chart')
        ]),
        dcc.Tab(label='Type Errors', children=[
            dcc.Graph(id='type-errors-chart')
        ]),
        dcc.Tab(label='Non-Standard Categories', children=[
            dcc.Graph(id='non-standard-chart')
        ])
    ])
])

@app.callback(
    [Output('missing-values-chart', 'figure'),
     Output('outliers-chart', 'figure'),
     Output('type-errors-chart', 'figure'),
     Output('non-standard-chart', 'figure')],
    [Input('missing-values-chart', 'relayoutData')]
)
def update_charts(_):
    missing_fig = px.bar(metrics_df[metrics_df['Issue Type'] == 'missing'],
                        x='Column', y='Count', title='Missing Values per Column')

    outliers_fig = px.bar(metrics_df[metrics_df['Issue Type'] == 'outliers'],
                         x='Column', y='Count', title='Outliers per Column')

    type_errors_fig = px.bar(metrics_df[metrics_df['Issue Type'] == 'type_errors'],
                            x='Column', y='Count', title='Type Errors per Column')

    non_standard_fig = px.bar(metrics_df[metrics_df['Issue Type'] == 'non_standard'],
                             x='Column', y='Count', title='Non-Standard Categories per Column')

    return missing_fig, outliers_fig, type_errors_fig, non_standard_fig

if __name__ == '__main__':
    app.run(debug=True)