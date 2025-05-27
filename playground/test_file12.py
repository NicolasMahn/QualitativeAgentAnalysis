import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html

# Load data
df = pd.read_csv('uploads/Employee.csv')

# 1. Overall gender distribution
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# 2. Gender distribution by city
city_gender = df.groupby(['City', 'Gender']).size().reset_index(name='Count')

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '20px'}, children=[
    html.H1("Employee Gender Distribution Analysis", style={'textAlign': 'center', 'color': '#2a3f5f'}),

    html.Div([
        dcc.Graph(
            id='overall-gender',
            figure=px.pie(gender_counts,
                         values='Count',
                         names='Gender',
                         title='Overall Gender Distribution',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            .update_traces(textposition='inside', textinfo='percent+label')
        )
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='city-gender',
            figure=px.bar(city_gender,
                         x='City',
                         y='Count',
                         color='Gender',
                         barmode='group',
                         title='Gender Distribution by City',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            .update_layout(xaxis_title='City', yaxis_title='Employee Count')
        )
    ], style={'width': '48%', 'display': 'inline-block'})
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)