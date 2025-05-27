import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html

# Load data
df = pd.read_csv('uploads/student_habits_performance.csv')

# Initialize Dash app
app = dash.Dash(__name__)

# Create visualizations
box_fig = px.box(df, x='gender', y='exam_score',
                title='Exam Score Distribution by Gender',
                color='gender',
                labels={'exam_score': 'Exam Score (%)'},
                height=400)

bar_fig = px.bar(df.groupby('gender', as_index=False)['exam_score'].mean(),
                x='gender', y='exam_score',
                title='Average Exam Score Comparison',
                labels={'exam_score': 'Average Score (%)'},
                color='gender',
                height=400)

# App layout
app.layout = html.Div([
    html.H1("Student Performance Analysis", style={'textAlign': 'center'}),

    html.Div([
        dcc.Graph(figure=box_fig)
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(figure=bar_fig)
    ], style={'width': '49%', 'display': 'inline-block'})
])

if __name__ == '__main__':
    app.run(debug=True)