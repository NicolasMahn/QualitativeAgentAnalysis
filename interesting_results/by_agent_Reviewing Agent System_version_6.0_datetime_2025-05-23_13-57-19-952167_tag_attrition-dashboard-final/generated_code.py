


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from dash import Dash, dcc, html, Input, Output, State
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load and prepare data
df = pd.read_csv("uploads/Employee.csv")

# 1. Data Exploration Visualizations
# Age vs Attrition
age_fig = px.box(df, x='LeaveOrNot', y='Age', 
                 title='<b>Age Distribution by Attrition Status</b>',
                 color='LeaveOrNot',
                 color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})

# Payment Tier vs Attrition
payment_fig = px.histogram(df, x='PaymentTier', color='LeaveOrNot',
                           title='<b>Attrition by Payment Tier</b>',
                           barmode='group',
                           category_orders={"PaymentTier": [1, 2, 3]},
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})

# Experience vs Attrition
exp_fig = px.violin(df, x='LeaveOrNot', y='ExperienceInCurrentDomain',
                    title='<b>Experience in Current Domain</b>',
                    color='LeaveOrNot',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})

# 2. Model Development
X = df[['Age', 'City', 'PaymentTier', 'Gender', 'EverBenched',
        'ExperienceInCurrentDomain', 'JoiningYear', 'Education']]
y = df['LeaveOrNot']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'PaymentTier', 'ExperienceInCurrentDomain', 'JoiningYear']),
        ('cat', OneHotEncoder(), ['City', 'Gender', 'EverBenched', 'Education'])
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
joblib.dump(model, 'output/attrition_model.joblib')

# 3. SHAP Explanation Setup
explainer = shap.Explainer(model.named_steps['classifier'], 
                          model.named_steps['preprocessor'].transform(X_train[:100]))

# 4. Dashboard Implementation
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Attrition Analysis & Prediction Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}),
    
    # Model Performance
    html.Div([
        html.Div(f"Model ROC-AUC: {roc_auc:.2f}", 
                style={'backgroundColor': '#3498db', 'color': 'white',
                       'padding': '15px', 'margin': '10px', 'borderRadius': '5px',
                       'fontSize': '18px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    
    # Visualizations Grid
    html.Div([
        dcc.Graph(figure=age_fig, style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(figure=payment_fig, style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(figure=exp_fig, style={'width': '100%'})
    ], style={'padding': '20px'}),
    
    # Prediction Interface
    html.Div([
        html.H3("Predict Attrition Risk", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        
        html.Div([
            # Left Column
            html.Div([
                html.Label("Age (20-40)", style={'fontWeight': 'bold'}),
                dcc.Input(id='age', type='number', min=20, max=40, value=30,
                         style={'marginBottom': '15px'}),
                
                html.Label("City", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='city', 
                            options=[{'label': c, 'value': c} for c in df['City'].unique()],
                            value='Bangalore',
                            style={'marginBottom': '15px'}),
                
                html.Label("Payment Tier", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='payment-tier',
                            options=[{'label': f'Tier {i}', 'value': i} for i in [1, 2, 3]],
                            value=2,
                            style={'marginBottom': '15px'}),
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
            
            # Right Column
            html.Div([
                html.Label("Gender", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='gender',
                            options=[{'label': g, 'value': g} for g in df['Gender'].unique()],
                            value='Male',
                            style={'marginBottom': '15px'}),
                
                html.Label("Ever Benched", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='benched',
                            options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}],
                            value='No',
                            style={'marginBottom': '15px'}),
                
                html.Label("Experience (Years)", style={'fontWeight': 'bold'}),
                dcc.Input(id='experience', type='number', min=0, max=10, value=2,
                         style={'marginBottom': '15px'}),
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'border': '1px solid #bdc3c7', 'borderRadius': '10px', 'padding': '20px'}),
        
        # Bottom Row
        html.Div([
            html.Label("Joining Year", style={'fontWeight': 'bold'}),
            dcc.Input(id='joining-year', type='number', min=2012, max=2022, value=2017,
                     style={'marginRight': '20px'}),
            
            html.Label("Education", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='education',
                        options=[{'label': e, 'value': e} for e in df['Education'].unique()],
                        value='Bachelors',
                        style={'width': '200px'})
        ], style={'marginTop': '20px', 'padding': '10px'}),
        
        html.Button('Predict', id='predict-btn', n_clicks=0,
                   style={'backgroundColor': '#2ecc71', 'color': 'white',
                          'padding': '15px 30px', 'border': 'none', 'borderRadius': '5px',
                          'marginTop': '20px', 'cursor': 'pointer'}),
        
        # Prediction Output
        html.Div([
            html.Div(id='prediction-output', 
                    style={'marginTop': '30px', 'padding': '20px',
                           'backgroundColor': '#ecf0f1', 'borderRadius': '10px',
                           'fontSize': '20px'}),
            dcc.Graph(id='shap-plot', style={'marginTop': '20px'})
        ])
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'})
])

# Prediction Callback
@app.callback(
    [Output('prediction-output', 'children'),
     Output('shap-plot', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [State('age', 'value'),
     State('city', 'value'),
     State('payment-tier', 'value'),
     State('gender', 'value'),
     State('benched', 'value'),
     State('experience', 'value'),
     State('joining-year', 'value'),
     State('education', 'value')]
)
def predict_attrition(n_clicks, age, city, payment, gender, benched, exp, join_year, edu):
    if n_clicks > 0:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame([[age, city, payment, gender, benched, exp, join_year, edu]],
                                   columns=X.columns)
            
            # Load model and predict
            model = joblib.load('output/attrition_model.joblib')
            prob = model.predict_proba(input_df)[0][1]
            
            # Generate SHAP explanation
            processed_input = model.named_steps['preprocessor'].transform(input_df)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            shap_values = explainer(processed_input)
            
            # Create SHAP waterfall plot
            fig = go.Figure(go.Waterfall(
                orientation="h",
                measure=["relative"] * len(shap_values[0].values),
                x=shap_values[0].values,
                y=feature_names,
                base=0,
                connector={"line": {"color": "rgb(63,63,63)"}}
            ))
            fig.update_layout(
                title="Feature Impact on Prediction",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Feature",
                margin=dict(l=150),
                showlegend=False
            )
            
            return f"Predicted Attrition Probability: {prob:.1%}", fig
            
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    return "Enter employee details and click 'Predict'", {}

if __name__ == '__main__':
    app.run_server(debug=True)


