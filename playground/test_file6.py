import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load and preprocess data
df = pd.read_csv('uploads/Employee.csv')
features = ['Age', 'City', 'PaymentTier', 'Gender', 'EverBenched',
           'ExperienceInCurrentDomain', 'JoiningYear', 'Education']
target = 'LeaveOrNot'

# Preprocessing pipeline
categorical_features = ['City', 'Gender', 'EverBenched', 'Education']
numeric_features = ['Age', 'PaymentTier', 'ExperienceInCurrentDomain', 'JoiningYear']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Model training
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced'))
])
model.fit(X_train, y_train)
joblib.dump(model, 'output/attrition_model.joblib')

# Create visualizations
corr_fig = px.scatter_matrix(df, dimensions=['Age', 'PaymentTier', 'ExperienceInCurrentDomain'],
                            color='LeaveOrNot', title='Attrition Correlation Matrix')

bench_fig = px.histogram(df, x='EverBenched', color='LeaveOrNot', barmode='group',
                        title='Attrition by Bench History',
                        labels={'EverBenched': 'Ever Benched?'})

# Create Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Employee Attrition Analysis & Prediction", style={'textAlign': 'center'}),

    html.Div([
        dcc.Graph(figure=corr_fig, style={'width': '49%', 'display': 'inline-block'}),
        dcc.Graph(figure=bench_fig, style={'width': '49%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Employee Details Input"),
        dcc.Dropdown(id='city', options=[{'label': c, 'value': c} for c in df['City'].unique()], placeholder='City'),
        dcc.Dropdown(id='gender', options=['Male', 'Female'], placeholder='Gender'),
        dcc.Dropdown(id='benched', options=['Yes', 'No'], placeholder='Ever Benched?'),
        dcc.Dropdown(id='education', options=['Bachelors', 'Masters'], placeholder='Education'),
        dcc.Input(id='age', type='number', placeholder='Age'),
        dcc.Input(id='payment', type='number', placeholder='Payment Tier (1-3)'),
        dcc.Input(id='experience', type='number', placeholder='Experience in Domain (years)'),
        dcc.Input(id='join_year', type='number', placeholder='Joining Year'),
        html.Button('Predict', id='predict-btn', n_clicks=0),
        html.Div(id='prediction-output')
    ]),
])

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('age', 'value'),
     State('city', 'value'),
     State('payment', 'value'),
     State('gender', 'value'),
     State('benched', 'value'),
     State('experience', 'value'),
     State('join_year', 'value'),
     State('education', 'value')]
)
def predict_attrition(n_clicks, age, city, payment, gender, benched, exp, join_year, edu):
    if all([age, city, payment, gender, benched, exp, join_year, edu]):
        input_data = pd.DataFrame([[
            age, city, payment, gender, benched, exp, join_year, edu
        ]], columns=features)
        model = joblib.load('output/attrition_model.joblib')
        prob = model.predict_proba(input_data)[0][1]
        return f"Attrition Probability: {prob*100:.1f}%"
    return "Please fill all fields to get prediction"

if __name__ == '__main__':
    app.run(debug=True)