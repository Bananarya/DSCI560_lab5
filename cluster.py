import mysql.connector
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
def cluster_and_visualization(db_config):
    conn = mysql.connector.connect(**db_config)
    query = "SELECT post_id,title_cleaned FROM reddit_posts_enhanced;"
    df = pd.read_sql(query, conn)
    conn.close()
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                tags=[str(i)]) for i,
                doc in enumerate(df["title_cleaned"])]

    model = Doc2Vec(vector_size=20,
                    min_count=2, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)


    document_vectors = [model.infer_vector(
        word_tokenize(doc.lower())) for doc in df["title_cleaned"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    kmeans.fit(document_vectors)

    labels = kmeans.labels_

    # Combine results into a DataFrame
    full_df = pd.DataFrame({'id':df["post_id"],'message': df["title_cleaned"], 'vector':document_vectors,'cluster': labels})

    X = np.vstack(full_df['vector'].to_numpy())

    # Reduce to 2D with PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    full_df['x'] = X_2d[:,0]
    full_df['y'] = X_2d[:,1]


    for c in sorted(full_df['cluster'].unique()):
        print(f"\nCluster {c}:")
        print(full_df[full_df['cluster']==c]['message'].head(5).to_string(index=False))
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = full_df.copy()

    top_n = 10  
    keywords_per_cluster = {}

    for cluster in sorted(df['cluster'].unique()):
        cluster_texts = df[df['cluster'] == cluster]['message']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(cluster_texts)

        avg_tfidf = np.mean(X.toarray(), axis=0)
        terms = vectorizer.get_feature_names_out()
        
        top_terms = [terms[i] for i in avg_tfidf.argsort()[::-1][:top_n]]
        keywords_per_cluster[cluster] = top_terms

    # Show results
    for cluster, keywords in keywords_per_cluster.items():
        print(f"Cluster {cluster}: {', '.join(keywords)}")

    X = np.vstack(full_df['vector'].to_numpy())
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    full_df['x'] = X_2d[:,0]
    full_df['y'] = X_2d[:,1]

    full_df['cluster_keywords'] = full_df['cluster'].map(keywords_per_cluster)

    fig = px.scatter(
        full_df,
        x='x',
        y='y',
        color='cluster',
        hover_name='message',               # shows post title
        hover_data={
            'cluster': True,               # show cluster ID
            'cluster_keywords': True,      # show cluster keywords
            'x': False,
            'y': False
        },
        title='Reddit Posts Clusters with Keywords',
        color_continuous_scale=px.colors.qualitative.T10
    )

    # Save to HTML
    fig.write_html("reddit_clusters_keywords.html")
