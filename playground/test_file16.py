import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import plotly.express as px

# Load and preprocess data
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv('uploads/student_habits_performance.csv')

    # Handle missing values
    df['parental_education_level'] = df['parental_education_level'].fillna('Unknown')

    # Convert categorical columns using label encoding
    cat_cols = ['gender', 'part_time_job', 'diet_quality',
                'parental_education_level', 'internet_quality',
                'extracurricular_participation']

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Standardize numerical features
    num_cols = ['age', 'study_hours_per_day', 'social_media_hours',
                'netflix_hours', 'attendance_percentage', 'sleep_hours',
                'exercise_frequency', 'mental_health_rating', 'exam_score']

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

df = load_and_preprocess()

# Sidebar controls
st.sidebar.header("Controls")

# Dimensionality reduction selection
dr_method = st.sidebar.selectbox(
    "Dimensionality Reduction Method:",
    ['PCA', 't-SNE']
)

# Clustering controls
st.sidebar.subheader("Clustering Settings")
cluster_algo = st.sidebar.selectbox(
    "Clustering Algorithm:",
    ['None', 'K-Means', 'DBSCAN', 'Agglomerative Clustering']
)

clusters = None
if cluster_algo == 'K-Means':
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
elif cluster_algo == 'DBSCAN':
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 5)
elif cluster_algo == 'Agglomerative Clustering':
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
    linkage = st.sidebar.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])

# Color mapping selection
color_by = st.sidebar.radio("Color points by:",
                           ['Cluster Labels', 'Dataset Attribute'])

selected_attribute = None
if color_by == 'Dataset Attribute':
    attributes = ['gender', 'part_time_job', 'diet_quality',
                 'internet_quality', 'parental_education_level',
                 'exam_score']
    selected_attribute = st.sidebar.selectbox("Select attribute", attributes)

# Perform dimensionality reduction
def perform_dr(method, data):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, perplexity=30)
    return reducer.fit_transform(data)

# Perform clustering
if cluster_algo != 'None':
    features = df.drop('student_id', axis=1)

    if cluster_algo == 'K-Means':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif cluster_algo == 'DBSCAN':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    elif cluster_algo == 'Agglomerative Clustering':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    clusters = clusterer.fit_predict(features)

# Create visualization
dr_results = perform_dr(dr_method, df.drop('student_id', axis=1))

plot_df = pd.DataFrame({
    'x': dr_results[:, 0],
    'y': dr_results[:, 1],
    'cluster': clusters if clusters is not None else np.zeros(len(df))
})

if color_by == 'Cluster Labels' and clusters is not None:
    color_data = plot_df['cluster']
    color_label = 'Cluster'
else:
    color_data = df[selected_attribute]
    color_label = selected_attribute

fig = px.scatter(plot_df, x='x', y='y',
                 color=color_data,
                 color_continuous_scale=px.colors.sequential.Viridis,
                 labels={'color': color_label},
                 title=f"{dr_method} Visualization with {color_label} Coloring")

fig.update_layout(xaxis_title=f"{dr_method} 1",
                 yaxis_title=f"{dr_method} 2")

st.plotly_chart(fig)

# Show cluster summary if applicable
if cluster_algo != 'None' and clusters is not None:
    st.subheader("Cluster Summary")
    cluster_df = df.copy()
    cluster_df['Cluster'] = clusters

    summary = cluster_df.groupby('Cluster').mean()
    st.dataframe(summary.style.background_gradient(cmap='Blues'))