import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from apyori import apriori
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer
import re
import time
import altair as alt
from streamlit_lottie import st_lottie
import requests
import json
import datetime

# Set up page configuration
st.set_page_config(
    page_title="Skill Mining",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)  # Added punkt to fix the error
    nltk.download('punkt_tab', quiet=True)  # Added to fix the LookupError
    return nltk.corpus.stopwords.words('english'), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4e54c8;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: #6366f1;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 0.5rem;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        opacity: 0.7;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("G:\Projects\job_skills.csv")
        return df.sample(n=1000, random_state=42)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide sample data if file doesn't exist
        return pd.DataFrame({
            'job_link': ['https://example.com/job1', 'https://example.com/job2'],
            'job_skills': ['python, data analysis, machine learning', 'sql, excel, tableau']
        })

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df['job_skills'] = df['job_skills'].fillna('')
    # Normalize skills - lowercase, strip whitespace, split by comma
    df['job_skills'] = df['job_skills'].apply(lambda x: ','.join([s.strip().lower() for s in x.split(',')]))
    # Extract job title from link if available
    df['job_title'] = df['job_link'].apply(lambda x: x.split('/')[-1].replace('-', ' ').title() if isinstance(x, str) else 'Unknown')
    return df

# Enhanced text preprocessing
@st.cache_data
def clean_and_tokenize(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = text.split()
    # Remove stopwords and short words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

@st.cache_data
def get_skills_from_text(text, custom_stopwords=None):
    if not text:
        return []
    
    words = text.lower().split(',')
    words = [word.strip() for word in words]
    
    if custom_stopwords:
        words = [word for word in words if word and word not in custom_stopwords and len(word) > 2]
    
    return words

@st.cache_data
def get_trending_skills(df, custom_stopwords=None):
    all_skills = []
    for skills in df['job_skills']:
        all_skills.extend(get_skills_from_text(skills, custom_stopwords))
    
    skill_count = Counter(all_skills)
    return skill_count.most_common(100)

@st.cache_data
def apply_clustering(df, n_clusters=4, algorithm='kmeans'):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english')
    X = vectorizer.fit_transform(df['job_skills'])
    
    # Clustering
    df_result = df.copy()
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_result['cluster'] = model.fit_predict(X)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=3)
        df_result['cluster'] = model.fit_predict(X)
    
    # Get top terms per cluster
    if algorithm == 'kmeans':
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        cluster_keywords = {}
        for i in range(n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]
            cluster_keywords[i] = top_terms
        
        df_result['cluster_keywords'] = df_result['cluster'].map(cluster_keywords)
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())
    df_result['x'] = coords[:, 0]
    df_result['y'] = coords[:, 1]
    
    return df_result

@st.cache_data
def skill_network(df, min_weight=2, custom_stopwords=None):
    # Create a graph of skill co-occurrences
    G = nx.Graph()
    
    for skills in df['job_skills']:
        skill_list = get_skills_from_text(skills, custom_stopwords)
        
        # Add nodes and edges
        for i, skill1 in enumerate(skill_list):
            if not G.has_node(skill1):
                G.add_node(skill1, count=1)
            else:
                # Check if 'count' attribute exists
                if 'count' in G.nodes[skill1]:
                    G.nodes[skill1]['count'] += 1
                else:
                    G.nodes[skill1]['count'] = 1
                
            for skill2 in skill_list[i+1:]:
                if G.has_edge(skill1, skill2):
                    G[skill1][skill2]['weight'] += 1
                else:
                    G.add_edge(skill1, skill2, weight=1)
    
    # Filter edges by minimum weight
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_weight]
    G.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G


@st.cache_data
def find_association_rules(df, min_support=0.01, min_confidence=0.3, custom_stopwords=None):
    transactions = []
    for skills in df['job_skills']:
        skill_list = get_skills_from_text(skills, custom_stopwords)
        if skill_list:
            transactions.append(skill_list)
    
    rules = list(apriori(
        transactions, 
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=1.1,
        max_length=3
    ))
    
    return rules

@st.cache_data
def generate_wordcloud(df, custom_stopwords=None):
    all_skills = []
    for skills in df['job_skills']:
        all_skills.extend(get_skills_from_text(skills, custom_stopwords))
    
    text = ' '.join(all_skills)
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=200,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    return wc

@st.cache_data
def calculate_skill_growth(df):
    # In a real app, this would use time-series data
    # Here we're simulating growth data
    top_skills = [skill for skill, _ in get_trending_skills(df)[:5]]
    
    # Create synthetic growth data
    growth_data = []
    now = datetime.datetime.now()
    
    for i in range(6):
        month = (now - datetime.timedelta(days=30*(5-i))).strftime('%Y-%m')
        for skill in top_skills:
            # Base frequency with some randomness
            base = np.random.randint(50, 100)
            # Add growth trend
            trend = base * (1 + (0.1 * i) + (np.random.random() * 0.05))
            growth_data.append({
                'skill': skill,
                'month': month,
                'count': int(trend)
            })
    
    return pd.DataFrame(growth_data)

@st.cache_data
def get_skill_recommendations(df, user_skills):
    # Create a simple recommendation system based on co-occurrence
    G = skill_network(df, min_weight=1)
    
    # Find skills that co-occur with user's skills
    recommendations = Counter()
    
    for skill in user_skills:
        if skill in G:
            for neighbor in G.neighbors(skill):
                if neighbor not in user_skills:
                    recommendations[neighbor] += G[skill][neighbor]['weight']
    
    # Return top recommendations
    return [skill for skill, _ in recommendations.most_common(5)]

def create_network_graph(G, layout='spring'):
    # Create a Plotly graph from a NetworkX graph
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:  # kamada_kawai
        pos = nx.kamada_kawai_layout(G)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        if 'weight' in edge[2]:
            edge_weights.append(edge[2]['weight'])
        else:
            edge_weights.append(1)
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if 'count' in G.nodes[node]:
            node_sizes.append(G.nodes[node]['count'])
        else:
            node_sizes.append(10)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[s * 2 for s in node_sizes],
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            line=dict(width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def frequent_pattern_mining(df, min_support=0.03, min_confidence=0.3, min_lift=1.1):
    # Extract association rules from skills
    transactions = []
    for skills in df['job_skills']:
        skill_list = get_skills_from_text(skills)
        if skill_list:
            transactions.append(skill_list)
    
    # Apply Apriori algorithm
    results = list(apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift
    ))
    
    # Format results
    rules = []
    for item in results:
        for stat in item.ordered_statistics:
            if len(stat.items_base) > 0 and len(stat.items_add) > 0:
                rules.append({
                    'antecedents': list(stat.items_base),
                    'consequents': list(stat.items_add),
                    'support': item.support,
                    'confidence': stat.confidence,
                    'lift': stat.lift
                })
    
    # Sort by lift
    rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
    return rules

def generate_salary_insights(df, trending_skills):
    # In a real app, this would use actual salary data
    # Here we're generating synthetic data
    salary_data = []
    
    for skill, count in trending_skills[:10]:
        # Generate synthetic salary ranges based on skill popularity
        base_salary = 50000 + (count * 100)
        min_salary = int(base_salary * (0.8 + np.random.random() * 0.1))
        max_salary = int(base_salary * (1.2 + np.random.random() * 0.3))
        avg_salary = int((min_salary + max_salary) / 2)
        
        salary_data.append({
            'skill': skill,
            'min_salary': min_salary,
            'max_salary': max_salary,
            'avg_salary': avg_salary
        })
    
    return pd.DataFrame(salary_data)

def load_lottie(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Main app
def main():
    # Load animation
    lottie_data = load_lottie("https://assets4.lottiefiles.com/packages/lf20_kU5CYg.json")
    
    # Title and description
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">LinkedIn Skill Mining</div>', unsafe_allow_html=True)
        if lottie_data:
            st_lottie(lottie_data, height=150, key="title_animation")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Discover trending skills, identify patterns, and gain insights from LinkedIn job data
    </div>
    """, unsafe_allow_html=True)
    
    # Load and process data
    with st.spinner("Loading and processing data..."):
        df = load_data()
        df = preprocess_data(df)
    
    # Custom stopwords
    custom_stopwords = {
        'of', 'the', 'and', 'to', 'in', 'for', 'with', 'a', 'an', 'on',
        'good', 'knowledge', 'basic', 'skills', 'experience', 'understanding', 'ability',
        'proficiency', 'proficient', 'expert', 'expertise', 'advanced', 'intermediate'
    }
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="subheader">Dashboard Controls</div>', unsafe_allow_html=True)
        
        # Data filters
        st.subheader("🔍 Filters")
        min_freq = st.slider("Minimum Skill Frequency", 1, 30, 5)
        
        # Clustering options
        st.subheader("🧩 Clustering")
        cluster_algorithm = st.selectbox(
            "Clustering Algorithm",
            options=["kmeans", "dbscan"],
            format_func=lambda x: "K-Means" if x == "kmeans" else "DBSCAN"
        )
        
        n_clusters = st.slider("Number of Clusters (K-Means)", 2, 10, 4)
        
        # Network options
        st.subheader("🌐 Network")
        min_edge_weight = st.slider("Minimum Connection Strength", 1, 10, 2)
        
        # Association rules
        st.subheader("🔗 Association Rules")
        min_support = st.slider("Minimum Support", 0.01, 0.2, 0.03, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05)
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.1, 0.1)
        
        # Additional options
        st.subheader("⚙️ Additional Options")
        show_salary = st.checkbox("Show Salary Insights", True)
        show_forecast = st.checkbox("Show Skill Forecasts", True)
        show_recommendations = st.checkbox("Show Skill Recommendations", True)
        
        # About
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Skill Mining Pro analyzes job listings to extract insights about the most in-demand skills and their relationships.")
    
    # Quick stats
    st.markdown('<div class="subheader">Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Job Listings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_skills = set()
        for skills in df['job_skills']:
            unique_skills.update(get_skills_from_text(skills, custom_stopwords))
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(unique_skills)}</div>
            <div class="stat-label">Unique Skills</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trending_skills = get_trending_skills(df, custom_stopwords)
        top_skill = trending_skills[0][0] if trending_skills else "N/A"
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{top_skill}</div>
            <div class="stat-label">Top Skill</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_skills_per_job = sum(len(get_skills_from_text(skills, custom_stopwords)) for skills in df['job_skills']) / len(df) if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_skills_per_job:.1f}</div>
            <div class="stat-label">Avg Skills/Job</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔥 Trending Skills", 
        "🧩 Skill Clusters", 
        "🌐 Skill Network", 
        "🔗 Pattern Mining", 
        "☁️ Skill Cloud",
        "📊 Skill Forecasts",
        "🧠 Skill Recommendations"
    ])
    
    # Tab 1: Trending Skills
    with tab1:
        st.markdown('<div class="subheader">Trending Skills Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trending_skills = get_trending_skills(df, custom_stopwords)
            skills_df = pd.DataFrame(trending_skills, columns=["Skill", "Frequency"])
            skills_df = skills_df[skills_df["Frequency"] >= min_freq]
            
            fig = px.bar(
                skills_df.head(20), 
                x="Frequency", 
                y="Skill", 
                orientation="h",
                color="Frequency",
                color_continuous_scale="viridis",
                title="Top 20 In-Demand Skills"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="Frequency",
                yaxis_title="",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Skill Distribution")
            
            # Calculate percentages for top 10 skills
            top_skills = skills_df.head(10)
            total = top_skills['Frequency'].sum()
            top_skills['Percentage'] = top_skills['Frequency'] / total * 100
            
            fig = px.pie(
                top_skills, 
                values='Percentage', 
                names='Skill',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.download_button(
                "📥 Download Skills Data",
                data=skills_df.to_csv(index=False),
                file_name="trending_skills.csv",
                mime="text/csv"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add salary insights if enabled
            if show_salary:
                st.markdown('<h3 class="subheader">Salary Insights by Skill</h3>', unsafe_allow_html=True)
                
                # Generate salary insights
                salary_df = generate_salary_insights(df, trending_skills)
                
                # Create salary range chart
                fig = go.Figure()
                
                for i, row in salary_df.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row['skill']],
                        y=[row['max_salary'] - row['min_salary']],
                        base=row['min_salary'],
                        name=row['skill'],
                        hovertemplate=f"<b>{row['skill']}</b><br>Range: ${row['min_salary']:,} - ${row['max_salary']:,}<br>Avg: ${row['avg_salary']:,}"
                    ))
                
                fig.update_layout(
                    title="Salary Ranges by Skill",
                    xaxis_title="Skill",
                    yaxis_title="Salary Range (USD)",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Skill Clusters
    with tab2:
        st.markdown('<div class="subheader">Skill Cluster Analysis</div>', unsafe_allow_html=True)
        
        # Apply clustering
        with st.spinner("Applying clustering algorithm..."):
            if cluster_algorithm == "kmeans":
                df_clustered = apply_clustering(df, n_clusters, "kmeans")
            else:
                df_clustered = apply_clustering(df, algorithm="dbscan")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Interactive scatter plot
            fig = px.scatter(
                df_clustered,
                x="x",
                y="y",
                color="cluster",
                color_continuous_scale="viridis" if cluster_algorithm == "dbscan" else px.colors.qualitative.Bold,
                hover_data=["job_title"],
                title="Job Clustering by Skills"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0.02)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Cluster Analysis")
            
            # Show cluster distribution
            cluster_counts = df_clustered['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            fig = px.bar(
                cluster_counts,
                x='Cluster',
                y='Count',
                color='Count',
                color_continuous_scale='viridis',
                title="Jobs per Cluster"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster keywords if using kmeans
            if cluster_algorithm == "kmeans" and 'cluster_keywords' in df_clustered.columns:
                selected_cluster = st.selectbox(
                    "Select cluster to view top skills",
                    options=sorted(df_clustered['cluster'].unique())
                )
                
                if selected_cluster is not None:
                    sample_row = df_clustered[df_clustered['cluster'] == selected_cluster].iloc[0]
                    if 'cluster_keywords' in sample_row:
                        st.write("#### Top Skills in Cluster")
                        keywords = sample_row['cluster_keywords']
                        for i, kw in enumerate(keywords):
                            st.write(f"{i+1}. {kw}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Skill Network
    with tab3:
        st.markdown('<div class="subheader">Skill Relationship Network</div>', unsafe_allow_html=True)
        
        with st.spinner("Building skill network..."):
            G = skill_network(df, min_edge_weight, custom_stopwords)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate node sizes based on occurrences
            node_sizes = [G.nodes[node]['count'] * 50 for node in G.nodes()]
            
            # Calculate edge widths based on weights
            edge_widths = [G[u][v]['weight'] / 2 for u, v in G.edges()]
            
            # Calculate node colors based on degree centrality
            centrality = nx.degree_centrality(G)
            node_colors = [centrality[node] for node in G.nodes()]
            
            # Position nodes using force-directed layout
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, cmap='viridis')
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='#888888')
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
            
            plt.axis('off')
            plt.tight_layout()
            
            st.pyplot(fig)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Network Metrics")
            
            # Calculate and display network metrics
            st.write(f"**Nodes:** {G.number_of_nodes()} skills")
            st.write(f"**Connections:** {G.number_of_edges()} relationships")
            
            if G.number_of_nodes() > 0:
                # Most central skills
                centrality = nx.degree_centrality(G)
                central_skills = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                
                st.write("**Most Connected Skills:**")
                for skill, score in central_skills[:5]:
                    st.write(f"- {skill} ({score:.3f})")
                
                # Calculate communities
                if G.number_of_nodes() > 1:
                    try:
                        communities = nx.community.greedy_modularity_communities(G)
                        st.write(f"**Skill Communities:** {len(communities)}")
                    except:
                        st.write("Could not calculate communities")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Pattern Mining
    with tab4:
        st.markdown('<div class="subheader">Skill Pattern Mining</div>', unsafe_allow_html=True)
        
        # Mine patterns
        rules = frequent_pattern_mining(df, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
        
        if rules:
            # Display rules
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h3 class="subheader">Association Rules</h3>', unsafe_allow_html=True)
                
                # Create rules table
                rules_df = pd.DataFrame([
                    {
                        'Antecedents': ', '.join(r['antecedents']),
                        'Consequents': ', '.join(r['consequents']),
                        'Support': r['support'],
                        'Confidence': r['confidence'],
                        'Lift': r['lift']
                    } for r in rules
                ])
                
                st.dataframe(rules_df, height=400)
            
            with col2:
                st.markdown('<h3 class="subheader">Pattern Metrics</h3>', unsafe_allow_html=True)
                
                # Create metrics visualization
                fig = px.scatter(
                    rules_df,
                    x='Support',
                    y='Confidence',
                    size='Lift',
                    hover_data=['Antecedents', 'Consequents'],
                    color='Lift',
                    color_continuous_scale='Viridis',
                    title="Pattern Metrics Visualization",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualize top rules
            st.markdown('<h3 class="subheader">Top Association Rules</h3>', unsafe_allow_html=True)
            
            # Create sankey diagram for top rules
            top_rules = rules[:min(10, len(rules))]
            
            # Prepare data for Sankey diagram
            nodes = []
            node_indices = {}
            links = []
            
            for rule in top_rules:
                for item in rule['antecedents'] + rule['consequents']:
                    if item not in node_indices:
                        node_indices[item] = len(nodes)
                        nodes.append(item)
                
                for source in rule['antecedents']:
                    for target in rule['consequents']:
                        links.append({
                            'source': node_indices[source],
                            'target': node_indices[target],
                            'value': rule['lift'] * 10  # Scale for visibility
                        })
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes
                ),
                link=dict(
                    source=[link['source'] for link in links],
                    target=[link['target'] for link in links],
                    value=[link['value'] for link in links]
                )
            )])
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No association rules found with the current parameters. Try adjusting the minimum support, confidence, or lift.")
    
    # Tab 5: Word Cloud
    with tab5:
        st.markdown('<div class="subheader">Skill Word Cloud</div>', unsafe_allow_html=True)
        
        with st.spinner("Generating word cloud..."):
            wordcloud = generate_wordcloud(df, custom_stopwords)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    # Tab 6: Skill Forecasts
    with tab6:
        if show_forecast:
            st.markdown('<h2 class="subheader">Skill Growth Forecasts</h2>', unsafe_allow_html=True)
            
            # Generate growth data
            growth_df = calculate_skill_growth(df)
            
            # Plot growth trends
            fig = px.line(
                growth_df,
                x='month',
                y='count',
                color='skill',
                title="Skill Growth Trends (6-Month Period)",
                labels={'month': 'Month', 'count': 'Frequency', 'skill': 'Skill'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast future trends
            st.markdown('<h3 class="subheader">Skill Growth Forecast</h3>', unsafe_allow_html=True)
            
            # Create a simple forecast (in a real app, this would use time series forecasting)
            skills = growth_df['skill'].unique()
            forecast_data = []
            
            now = datetime.datetime.now()
            for skill in skills:
                skill_data = growth_df[growth_df['skill'] == skill]
                last_count = skill_data['count'].iloc[-1]
                
                # Simple growth projection (in a real app, use ARIMA or other forecasting methods)
                for i in range(1, 4):
                    month = (now + datetime.timedelta(days=30*i)).strftime('%Y-%m')
                    growth_factor = 1 + (np.random.normal(0.05, 0.02) * i)
                    forecast_data.append({
                        'skill': skill,
                        'month': month,
                        'count': int(last_count * growth_factor),
                        'type': 'forecast'
                    })
            
            # Combine historical and forecast data
            historical_data = growth_df.copy()
            historical_data['type'] = 'historical'
            forecast_df = pd.concat([historical_data, pd.DataFrame(forecast_data)])
            
            # Plot forecast
            fig = px.line(
                forecast_df,
                x='month',
                y='count',
                color='skill',
                line_dash='type',
                title="Skill Growth Forecast (Next 3 Months)",
                labels={'month': 'Month', 'count': 'Frequency', 'skill': 'Skill', 'type': 'Data Type'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth rate calculation
            st.markdown('<h3 class="subheader">Skill Growth Rates</h3>', unsafe_allow_html=True)
            
            # Calculate growth rates
            growth_rates = []
            for skill in skills:
                skill_data = historical_data[historical_data['skill'] == skill]
                if len(skill_data) >= 2:
                    first_count = skill_data['count'].iloc[0]
                    last_count = skill_data['count'].iloc[-1]
                    growth_rate = ((last_count / first_count) - 1) * 100
                    growth_rates.append({
                        'skill': skill,
                        'growth_rate': growth_rate
                    })
            
            growth_rates_df = pd.DataFrame(growth_rates)
            growth_rates_df = growth_rates_df.sort_values('growth_rate', ascending=False)
            
            # Plot growth rates
            fig = px.bar(
                growth_rates_df,
                x='skill',
                y='growth_rate',
                color='growth_rate',
                color_continuous_scale='RdYlGn',
                title="Skill Growth Rates (%)",
                labels={'skill': 'Skill', 'growth_rate': 'Growth Rate (%)'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Skill forecasts are disabled. Enable them in the sidebar to view growth projections.")
    
    # Tab 7: Skill Recommendations
    with tab7:
        if show_recommendations:
            st.markdown('<h2 class="subheader">Skill Recommendations</h2>', unsafe_allow_html=True)
            
            # Input for skills
            st.markdown('<h3 class="subheader">Enter Your Skills</h3>', unsafe_allow_html=True)
            
            # Multi-select for skills
            all_skills = [skill for skill, _ in trending_skills[:50]]
            selected_skills = st.multiselect("Select your current skills", all_skills)
            
            if selected_skills:
                # Get recommendations
                recommendations = get_skill_recommendations(df, selected_skills)
                
                st.markdown('<h3 class="subheader">Recommended Skills</h3>', unsafe_allow_html=True)
                
                if recommendations:
                    # Create columns for recommendations
                    cols = st.columns(3)
                    for i, skill in enumerate(recommendations):
                        with cols[i % 3]:
                            st.markdown(f'<div class="card">', unsafe_allow_html=True)
                            st.markdown(f'<div class="stat-value">{skill}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="stat-label">Recommendation #{i+1}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Create skill path visualization
                    st.markdown('<h3 class="subheader">Skill Path Visualization</h3>', unsafe_allow_html=True)
                    
                    # Create nodes for current and recommended skills
                    nodes = selected_skills + recommendations[:5]
                    links = []
                    
                    # Create links from current to recommended skills
                    for source in selected_skills:
                        for target in recommendations[:5]:
                            links.append({'source': source, 'target': target})
                    
                    # Create network
                    G = nx.DiGraph()
                    for node in nodes:
                        G.add_node(node)
                    
                    for link in links:
                        G.add_edge(link['source'], link['target'])
                    
                    # Create network visualization
                    fig = create_network_graph(G, layout='spring')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No recommendations found for the selected skills.")
            else:
                st.info("Select your current skills to get recommendations.")
        else:
            st.info("Skill recommendations are disabled. Enable them in the sidebar to view personalized suggestions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
    <p>Skill Mining Pro Dashboard | Created with Streamlit | Data Last Updated: April 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
