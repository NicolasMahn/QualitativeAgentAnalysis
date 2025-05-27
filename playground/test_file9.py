import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# Load and prepare data
df = pd.read_csv('uploads/Employee.csv')

# Education categorization
df['Education Level'] = df['Education'].apply(lambda x:
    'PhD' if 'phd' in x.lower() else
    'Master\'s' if 'master' in x.lower() else
    'Bachelor\'s' if 'bachelor' in x.lower() else
    'Other'
)

# Experience categorization
df['Experience Group'] = pd.cut(df['ExperienceInCurrentDomain'],
                               bins=[-1, 2, 5, 20],
                               labels=['0-2 Years', '3-5 Years', '6+ Years'])

# New Visualizations
# 1. Experience vs Payment Tier
fig_experience = px.histogram(df, x='Experience Group', color='PaymentTier',
                            barmode='group', title='Payment Tier Distribution by Experience Level')

# 2. Education-Payment relationship faceted by experience
fig_education_stratified = px.box(df, x='Education Level', y='PaymentTier',
                                 facet_col='Experience Group', color='Education Level',
                                 title='Education vs Payment Tier Stratified by Experience')

# 3. Marginal relationship visualization
fig_marginal = px.scatter(df, x='ExperienceInCurrentDomain', y='PaymentTier',
                         color='Education Level', trendline='lowess',
                         title='Payment Tier Relationships with Experience & Education',
                         marginal_x='histogram', marginal_y='violin')

# Update Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Compensation Analysis Dashboard"),

    html.Div([
        html.H2("Experience Analysis"),
        dcc.Graph(figure=fig_experience),
        dcc.Graph(figure=fig_marginal)
    ]),

    html.Div([
        html.H2("Education Analysis (Experience Adjusted)"),
        dcc.Graph(figure=fig_education_stratified)
    ]),

    html.Div([
        html.H4("Data Summary"),
        html.P(f"Total Employees: {len(df)}"),
        html.P(f"Experience Range: {df['ExperienceInCurrentDomain'].min()} - {df['ExperienceInCurrentDomain'].max()} Years"),
        html.P(f"Payment Tier Distribution: {dict(df['PaymentTier'].value_counts())}")
    ])
])

if __name__ == '__main__':
    app.run(debug=True)