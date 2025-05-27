import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html

# Load data
df = pd.read_csv('uploads/Employee.csv')

# Standardize education levels (example: handle case variations)
df['Education'] = df['Education'].str.strip().str.title()

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Employee Education vs. Payment Tier Analysis"),

    # Box plot: Payment Tier distribution by Education
    dcc.Graph(
        id='box-plot',
        figure=px.box(
            df, x='Education', y='PaymentTier',
            title='Payment Tier Distribution by Education Level',
            labels={'PaymentTier': 'Payment Tier', 'Education': 'Education Level'},
            color='Education'
        )
    ),

    # Bar chart: Average Payment Tier per Education
    dcc.Graph(
        id='bar-chart',
        figure=px.bar(
            df.groupby('Education', as_index=False)['PaymentTier'].mean(),
            x='Education', y='PaymentTier',
            title='Average Payment Tier by Education Level',
            labels={'PaymentTier': 'Average Payment Tier'}
        )
    )
])

if __name__ == '__main__':
    app.run(debug=True)