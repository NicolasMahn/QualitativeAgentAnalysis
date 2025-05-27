# Input files: ['/app/agent_files/simple_agent_system/uploads/Employee.csv']




import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import joblib
import shap
import numpy as np

# Load pre-trained model
model = joblib.load('output/attrition_model.joblib')
df = pd.read_csv("uploads/Employee.csv")

# Create SHAP explainer
preprocessed_data = model.named_steps['preprocessor'].transform(df)
explainer = shap.TreeExplainer(model.named_steps['classifier'])

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Employee Attrition Analysis Dashboard"),

    # Existing visualizations and input components remain unchanged

    # Prediction Output Section
    html.Div([
        html.Div(id='prediction-output', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='shap-plot', style={'width': '48%', 'display': 'inline-block'})
    ])
])

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
def predict_and_explain(n_clicks, age, city, payment, gender, benched, exp, year, edu):
    if n_clicks == 0:
        raise PreventUpdate

    # Create input DataFrame
    input_df = pd.DataFrame([[edu, year, city, payment, age, gender, benched, exp]],
        columns=['Education','JoiningYear','City','PaymentTier','Age',
                'Gender','EverBenched','ExperienceInCurrentDomain'])

    # Get prediction probability
    proba = model.predict_proba(input_df)[0][1]

    # Calculate SHAP values
    processed_input = model.named_steps['preprocessor'].transform(input_df)
    shap_values = explainer.shap_values(processed_input)[1]

    # Get feature names after preprocessing
    cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    num_features = numerical_features = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']
    all_features = num_features + list(cat_features)

    # Create SHAP visualization
    fig = px.bar(
        x=shap_values[0],
        y=all_features,
        orientation='h',
        title='Feature Impact on Prediction',
        labels={'x': 'Impact Value', 'y': 'Feature'},
        color=np.where(shap_values[0] > 0, 'Positive', 'Negative'),
        color_discrete_map={'Positive':'#4d9221', 'Negative':'#c51b7d'}
    )
    fig.update_layout(showlegend=False)

    prediction_output = html.Div([
        html.H4(f"Predicted Attrition Probability: {proba:.1%}"),
        html.P("Key factors influencing this prediction:"),
        html.Ul([
            html.Li(f"{feature}: {value:.2f} impact")
            for feature, value in zip(all_features, shap_values[0])
            if abs(value) > 0.01  # Only show significant contributors
        ])
    ])

    return prediction_output, fig

# Rest of the existing code remains unchanged
if __name__ == '__main__':
    app.run(debug=True)