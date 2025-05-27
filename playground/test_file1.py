import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv('../old_test_task_1/data/Employee.csv')

# Map Education to three levels: 'Bachelors', 'Masters', 'PhD', others as 'Others'
def map_education_level(val):
    val_lower = val.lower()
    if 'phd' in val_lower:
        return 'PhD'
    elif 'master' in val_lower:
        return 'Masters'
    elif 'bachelor' in val_lower:
        return 'Bachelors'
    else:
        return 'Others'

df['Education_Level'] = df['Education'].apply(map_education_level)

# Create "Experience in Current Domain" bins for categorical visualization (for plot)
exp_bins = [0, 2, 5, 10, np.inf]
exp_labels = ['0-2', '3-5', '6-10', '10+']
df['Experience_Bin'] = pd.cut(df['ExperienceInCurrentDomain'], bins=exp_bins, labels=exp_labels, right=True, include_lowest=True)

# Relationship: Summary statistics of PaymentTier by Experience_Bin
experience_payment = df.groupby('Experience_Bin')['PaymentTier'].describe()

# Encode 'Education_Level' as ordinal for regression
education_order = ['Bachelors', 'Masters', 'PhD', 'Others']
df['Education_Level_Code'] = pd.Categorical(df['Education_Level'], categories=education_order, ordered=True).codes

# Linear regression: PaymentTier ~ Education_Level_Code + ExperienceInCurrentDomain
X = df[['Education_Level_Code', 'ExperienceInCurrentDomain']]
y = df['PaymentTier']
lr_model = LinearRegression()
lr_model.fit(X, y)

# Compute adjusted payment tier controlling for experience (regression prediction)
df['Adjusted_PaymentTier'] = lr_model.predict(X)

# Plot: Experience_Bin vs PaymentTier (box plot)
fig_exp_payment = px.box(
    df, x='Experience_Bin', y='PaymentTier',
    category_orders={'Experience_Bin': exp_labels},
    labels={'Experience_Bin': 'Experience in Current Domain (years)', 'PaymentTier': 'Payment Tier'},
    title='Payment Tier Distribution by Experience Level'
)

# Plot: Education_Level vs Adjusted_PaymentTier (box plot) controlling for Experience
fig_edu_adj_payment = px.box(
    df, x='Education_Level', y='Adjusted_PaymentTier',
    category_orders={'Education_Level': education_order},
    labels={'Education_Level': 'Education Level', 'Adjusted_PaymentTier': 'Adjusted Payment Tier (controlling for Experience)'},
    title='Adjusted Payment Tier by Education Level (controlled for Experience)'
)

# Build the Dash app with the new visualizations
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Data Analysis Dashboard", style={'textAlign': 'center'}),

    html.H2("1. Experience in Current Domain vs Payment Tier"),
    dcc.Graph(id='exp-payment-box', figure=fig_exp_payment),

    html.H2("2. Education vs Payment Tier (controlled for Experience)"),
    dcc.Graph(id='edu-adjusted-payment-box', figure=fig_edu_adj_payment),

    html.H3("Regression Model Coefficients:"),
    html.Pre(f"Intercept: {lr_model.intercept_:.4f}\n"
             f"Education Level Coef: {lr_model.coef_[0]:.4f}\n"
             f"Experience Coef: {lr_model.coef_[1]:.4f}")
])

# This app variable can be used to run the server locally or on a deployment platform

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Port modified for local testing