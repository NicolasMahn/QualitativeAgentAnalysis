import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import numpy as np

# Load data
df = pd.read_csv('uploads/unclean_smartwatch_health_data.csv')

# 1. Data Quality Analysis
# Initialize issues DataFrame
columns = df.columns
issues = pd.DataFrame(index=columns, columns=['Missing', 'Outliers', 'Type Errors', 'Non-Std Categories']).fillna(0)

# Missing Values
issues['Missing'] = df.isnull().sum().values

# Outliers (using IQR method)
numeric_cols = ['User ID', 'Heart Rate (BPM)', 'Blood Oxygen Level (%)', 'Step Count']
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    issues.loc[col, 'Outliers'] = outliers

# Type Errors
# Sleep Duration
sleep_type_errors = df['Sleep Duration (hours)'].apply(lambda x: not str(x).replace('.','',1).isdigit() and x != 'nan').sum()
issues.loc['Sleep Duration (hours)', 'Type Errors'] = sleep_type_errors

# Stress Level
stress_type_errors = df['Stress Level'].apply(lambda x: not str(x).isdigit() if pd.notnull(x) else False).sum()
issues.loc['Stress Level', 'Type Errors'] = stress_type_errors

# Non-Standard Categories
activity_categories = ['Highly Active', 'Active', 'Sedentary']
non_std = df['Activity Level'].isin(activity_categories) | df['Activity Level'].isna()
issues.loc['Activity Level', 'Non-Std Categories'] = (~non_std).sum()

# Duplicates
duplicates = df.duplicated().sum()

# 2. Dash Dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Smartwatch Health Data Quality Dashboard"),

    html.Div([
        html.H3(f"Total Duplicate Rows: {duplicates}"),
    ], style={'margin': '20px'}),

    dcc.Tabs([
        dcc.Tab(label='Missing Values', children=[
            dcc.Graph(
                figure=px.bar(issues, x=issues.index, y='Missing',
                             title='Missing Values per Column')
            )
        ]),

        dcc.Tab(label='Outliers', children=[
            dcc.Graph(
                figure=px.bar(issues[issues['Outliers'] > 0], x=issues.index, y='Outliers',
                             title='Outliers in Numeric Columns')
            )
        ]),

        dcc.Tab(label='Type Errors', children=[
            dcc.Graph(
                figure=px.bar(issues[issues['Type Errors'] > 0], x=issues.index, y='Type Errors',
                             title='Type Errors per Column')
            )
        ]),

        dcc.Tab(label='Non-Std Categories', children=[
            dcc.Graph(
                figure=px.bar(issues[issues['Non-Std Categories'] > 0], x=issues.index, y='Non-Std Categories',
                             title='Non-Standard Categories in Activity Level')
            )
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)