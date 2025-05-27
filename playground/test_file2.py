import pandas as pd
import numpy as np
import re
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess dataset
df = pd.read_csv('../old_test_task_1/data/Employee.csv')

# Identify target column: check if 'Leave or Not' exists with correct name, else try alternatives
target_col = None
for col in ['Leave or Not', 'Leave_or_Not', 'leave_or_not', 'Leave_or_not', 'leave or not']:
    if col in df.columns:
        target_col = col
        break
if target_col is None:
    raise ValueError("Could not find 'Leave or Not' target column in dataset.")

# Convert target to numeric binary if needed
if df[target_col].dtype == object:
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 'yes':1, 'no':0})

# Variables for EDA
num_vars = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain']
cat_vars = ['Ever Benched', 'Gender']

# Point-biserial correlation for numeric variables
from scipy.stats import pointbiserialr
corrs = {}
for col in num_vars:
    try:
        corr, pval = pointbiserialr(df[target_col], df[col])
        corrs[col] = (corr, pval)
    except Exception:
        corrs[col] = (np.nan, np.nan)

# Categorical variables analysis: proportion leaving per category
cat_stats = {}
for col in cat_vars:
    ctab = pd.crosstab(df[col], df[target_col], normalize='index').reset_index()
    cat_stats[col] = ctab

# Select features for modeling
features = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain', 'Ever Benched', 'Gender', 'City', 'Joining Year', 'Education']

# Identify numeric and categorical features
numeric_features = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain', 'Joining Year']
categorical_features = ['Ever Benched', 'Gender', 'City', 'Education']

# Fill missing numeric with median
for col in numeric_features:
    if col in df.columns and df[col].isna().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical with 'Unknown'
for col in categorical_features:
    if col in df.columns and df[col].isna().sum() > 0:
        df[col].fillna('Unknown', inplace=True)

# Use subset of df for features (handle if some columns not present)
X = df[features].copy()

# Map Education levels similar to prior
def map_education_level(val):
    val_lower = str(val).lower()
    if 'phd' in val_lower:
        return 'PhD'
    elif 'master' in val_lower:
        return 'Masters'
    elif 'bachelor' in val_lower:
        return 'Bachelors'
    else:
        return 'Others'
if 'Education' in X.columns:
    X['Education'] = X['Education'].apply(map_education_level)

y = df[target_col].values

# Build preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Logistic Regression Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
model.fit(X, y)

# Prepare user input dropdown options for categorical fields
unique_cities = sorted(df['City'].dropna().unique()) if 'City' in df.columns else []
unique_education = sorted(df['Education'].dropna().unique()) if 'Education' in df.columns else []
ever_benched_options = [{'label': v, 'value': v} for v in sorted(df['Ever Benched'].dropna().unique())] if 'Ever Benched' in df.columns else []
gender_options = [{'label': v, 'value': v} for v in sorted(df['Gender'].dropna().unique())] if 'Gender' in df.columns else []

# Prepare box plots (numeric variables vs target)
def create_num_boxplot(col):
    if col in df.columns:
        fig = px.box(df, x=target_col, y=col, points='all',
                     labels={target_col: 'Leave or Not', col: col},
                     title=f'Distribution of {col} by Leave or Not')
        fig.update_layout(template='plotly_white')
        return fig
    return {}

# Prepare bar plots (categorical variable leave rate)
def create_cat_barplot(col):
    if col in cat_stats:
        df_stats = cat_stats[col]
        if 1 in df_stats.columns:
            df_stats['leave_rate'] = df_stats[1]
        else:
            df_stats['leave_rate'] = 0
        fig = px.bar(df_stats, x=col, y='leave_rate',
                     labels={col: col, 'leave_rate': 'Proportion Leaving'},
                     title=f'Proportion of Leaving by {col}')
        fig.update_layout(template='plotly_white')
        return fig
    return {}

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Attrition Analysis and Prediction Dashboard",
            style={'textAlign': 'center'}),

    html.Div([
        html.H3("Exploratory Data Analysis: Numeric Variables vs Leave or Not"),
        dcc.Tabs(id='num-tabs', value=num_vars[0], children=[
            dcc.Tab(label=col, value=col) for col in num_vars
        ]),
        dcc.Graph(id='num-box-plot')
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.H3("Exploratory Data Analysis: Categorical Variables Leave Rate"),
        dcc.Tabs(id='cat-tabs', value=cat_vars[0], children=[
            dcc.Tab(label=col, value=col) for col in cat_vars
        ]),
        dcc.Graph(id='cat-bar-plot'),
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.H3("Predict Probability of Employee Leaving"),
    html.Div([
        html.Div([
            html.Label("Age:"),
            dcc.Input(id='input-age', type='number', value=30, min=18, max=70, step=1)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Payment Tier (1=Low, 2=Medium, 3=High):"),
            dcc.Dropdown(id='input-payment-tier',
                         options=[{'label': str(i), 'value': i} for i in sorted(df['PaymentTier'].dropna().unique())] if 'PaymentTier' in df.columns else [],
                         value=int(df['PaymentTier'].mode()[0]) if 'PaymentTier' in df.columns else 1)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Experience in Current Domain (years):"),
            dcc.Input(id='input-experience', type='number', value=5, min=0, max=50, step=1)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Ever Benched:"),
            dcc.Dropdown(id='input-ever-benched',
                         options=ever_benched_options,
                         value=ever_benched_options[0]['value'] if ever_benched_options else None)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Gender:"),
            dcc.Dropdown(id='input-gender',
                         options=gender_options,
                         value=gender_options[0]['value'] if gender_options else None)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("City:"),
            dcc.Dropdown(id='input-city',
                         options=[{'label': city, 'value': city} for city in unique_cities],
                         value=unique_cities[0] if unique_cities else None)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Joining Year:"),
            dcc.Input(id='input-join-year', type='number', value=2015, min=1990, max=2025, step=1)
        ], style={'padding': '10px'}),
        html.Div([
            html.Label("Education:"),
            dcc.Dropdown(id='input-education',
                         options=[{'label': edu, 'value': edu} for edu in unique_education],
                         value=unique_education[0] if unique_education else None)
        ], style={'padding': '10px'}),
        html.Button('Predict Attrition Probability', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output',
                 style={'padding': '20px', 'fontSize': '20px', 'color': 'blue', 'fontWeight': 'bold'})
    ], style={'border': '1px solid #ccc', 'padding': '15px', 'maxWidth': '600px', 'margin': 'auto'})
])

# Callbacks for EDA numeric tabs
@app.callback(
    Output('num-box-plot', 'figure'),
    Input('num-tabs', 'value')
)
def update_num_boxplot(selected_col):
    return create_num_boxplot(selected_col)

# Callbacks for EDA categorical tabs
@app.callback(
    Output('cat-bar-plot', 'figure'),
    Input('cat-tabs', 'value')
)
def update_cat_barplot(selected_col):
    return create_cat_barplot(selected_col)

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-age', 'value'),
    State('input-payment-tier', 'value'),
    State('input-experience', 'value'),
    State('input-ever-benched', 'value'),
    State('input-gender', 'value'),
    State('input-city', 'value'),
    State('input-join-year', 'value'),
    State('input-education', 'value')
)
def predict_attrition(n_clicks, age, payment_tier, experience,
                      ever_benched, gender, city, join_year, education):
    if n_clicks is None or n_clicks == 0:
        return ""
    # Create a DataFrame with user input
    input_dict = {
        'Age': [age],
        'PaymentTier': [payment_tier],
        'ExperienceInCurrentDomain': [experience],
        'Ever Benched': [ever_benched],
        'Gender': [gender],
        'City': [city],
        'Joining Year': [join_year],
        'Education': [education]
    }
    input_df = pd.DataFrame(input_dict)
    # Map Education as in dataset
    input_df['Education'] = input_df['Education'].apply(map_education_level)
    # Some values might be missing in options, fill with 'Unknown' to allow one-hot encoding
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col].fillna('Unknown', inplace=True)
    # Predict probability
    try:
        prob = model.predict_proba(input_df)[0][1]
        return f"Predicted Probability of Leaving: {prob:.2%}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Port modified