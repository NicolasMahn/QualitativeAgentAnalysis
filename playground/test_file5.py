import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load and preprocess dataset
df = pd.read_csv('../old_test_task_1/data/Employee.csv')

# Define features and target
X = df[['Age', 'City', 'PaymentTier', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain', 'JoiningYear', 'Education']]
y = df['LeaveOrNot']

# Preprocessing pipeline
categorical_features = ['City', 'Gender', 'Education', 'EverBenched']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', ['Age', 'PaymentTier', 'ExperienceInCurrentDomain', 'JoiningYear'])
    ])

# Create and train model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression())
])
model.fit(X, y)

# Save model and preprocessor
joblib.dump(model, 'attrition_model.pkl')

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Attrition Analysis Dashboard", style={'textAlign': 'center'}),

    # Visualization Section
    html.Div([
        dcc.Dropdown(id='var-selector',
                     options=[
                         {'label': 'Age', 'value': 'Age'},
                         {'label': 'Experience', 'value': 'ExperienceInCurrentDomain'},
                         {'label': 'Payment Tier', 'value': 'PaymentTier'}
                     ],
                     value='Age',
                     style={'width': '50%', 'margin': '10px'}),
        dcc.Graph(id='main-plot')
    ], style={'padding': '20px'}),

    # Prediction Interface
    html.Div([
        html.H3("Predict Attrition Risk"),
        html.Div([
            dcc.Input(id='age', type='number', placeholder='Age', style={'margin': '5px'}),
            dcc.Dropdown(id='city', options=[{'label': c, 'value': c} for c in df['City'].unique()],
                        placeholder='City', style={'margin': '5px'}),
            dcc.Dropdown(id='gender', options=[{'label': g, 'value': g} for g in df['Gender'].unique()],
                        placeholder='Gender', style={'margin': '5px'})
        ], style={'columnCount': 3}),

        html.Div([
            dcc.Dropdown(id='education', options=[{'label': e, 'value': e} for e in df['Education'].unique()],
                        placeholder='Education', style={'margin': '5px'}),
            dcc.Input(id='experience', type='number', placeholder='Experience (Years)', style={'margin': '5px'}),
            dcc.Dropdown(id='benched', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                        placeholder='Ever Benched?', style={'margin': '5px'})
        ], style={'columnCount': 3}),

        html.Div([
            dcc.Input(id='payment-tier', type='number', placeholder='Payment Tier (1-3)', min=1, max=3,
                     style={'margin': '5px'}),
            dcc.Input(id='joining-year', type='number', placeholder='Joining Year',
                     style={'margin': '5px'})
        ], style={'columnCount': 2}),

        html.Button('Predict', id='predict-btn', style={'margin': '10px'}),
        html.Div(id='prediction-output', style={'fontSize': '20px', 'padding': '20px'})
    ], style={'border': '1px solid #ddd', 'padding': '20px', 'margin': '20px'})
])

@app.callback(
    Output('main-plot', 'figure'),
    [Input('var-selector', 'value')]
)
def update_plot(selected_var):
    if selected_var in ['Age', 'ExperienceInCurrentDomain']:
        fig = px.box(df, x='LeaveOrNot', y=selected_var,
                    title=f"{selected_var} vs Attrition")
    else:
        agg_data = df.groupby([selected_var, 'LeaveOrNot']).size().reset_index(name='Count')
        fig = px.bar(agg_data, x=selected_var, y='Count', color='LeaveOrNot',
                    title=f"Attrition by {selected_var}", barmode='group')
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('age', 'value'),
     State('city', 'value'),
     State('gender', 'value'),
     State('education', 'value'),
     State('experience', 'value'),
     State('benched', 'value'),
     State('payment-tier', 'value'),
     State('joining-year', 'value')]
)
def predict_attrition(_, age, city, gender, education, exp, benched, payment, joining_year):
    if None in [age, city, gender, education, exp, benched, payment, joining_year]:
        return "Please fill all fields"

    input_df = pd.DataFrame([[age, city, payment, gender, benched, exp, joining_year, education]],
                            columns=['Age','City','PaymentTier','Gender','EverBenched',
                                    'ExperienceInCurrentDomain','JoiningYear','Education'])

    model = joblib.load('attrition_model.pkl')
    prob = model.predict_proba(input_df)[0][1]

    return f"Attrition Probability: {prob*100:.1f}%"

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Port modified