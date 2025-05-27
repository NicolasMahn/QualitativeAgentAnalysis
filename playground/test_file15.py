import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load and preprocess data
df = pd.read_csv('uploads/student_habits_performance.csv')

# Select numerical features
numerical_features = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendance_percentage', 'sleep_hours', 'exercise_frequency',
    'mental_health_rating', 'exam_score'
]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_features])

# Precompute dimensionality reductions
## PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)
df['PCA1'] = pca_results[:, 0]
df['PCA2'] = pca_results[:, 1]

## t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X_scaled)
df['TSNE1'] = tsne_results[:, 0]
df['TSNE2'] = tsne_results[:, 1]

## UMAP
umap_reducer = umap.UMAP(random_state=42)
umap_results = umap_reducer.fit_transform(X_scaled)
df['UMAP1'] = umap_results[:, 0]
df['UMAP2'] = umap_results[:, 1]

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Student Habits Performance Explorer",
            style={'textAlign': 'center'}),

    html.Div([
        dcc.Dropdown(
            id='dr-method',
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 't-SNE', 'value': 'TSNE'},
                {'label': 'UMAP', 'value': 'UMAP'}
            ],
            value='PCA',
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'padding': '20px'}),

    dcc.Graph(id='dr-plot', style={'height': '80vh'})
])

@app.callback(
    Output('dr-plot', 'figure'),
    [Input('dr-method', 'value')]
)
def update_graph(selected_method):
    method_map = {
        'PCA': ['PCA1', 'PCA2'],
        'TSNE': ['TSNE1', 'TSNE2'],
        'UMAP': ['UMAP1', 'UMAP2']
    }

    fig = px.scatter(
        df,
        x=method_map[selected_method][0],
        y=method_map[selected_method][1],
        color='gender',
        hover_data=['student_id', 'age', 'exam_score'],
        title=f'{selected_method} Visualization of Student Data',
        labels={
            method_map[selected_method][0]: f'{selected_method} Dimension 1',
            method_map[selected_method][1]: f'{selected_method} Dimension 2',
            'gender': 'Gender'
        },
        color_discrete_map={'Male': '#636EFA', 'Female': '#EF553B'}
    )

    fig.update_layout(
        transition_duration=500,
        plot_bgcolor='rgba(240,240,240,0.9)',
        paper_bgcolor='rgba(240,240,240,0.9)'
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)