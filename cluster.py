import os, pickle
from pathlib import Path
import mysql.connector
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

def tok(s):
    return (s or "").lower().split()

def train_and_save(db_config, n_clusters=3, outdir="artifacts"):
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql("""
        SELECT post_id, COALESCE(NULLIF(title_cleaned,''), title) AS message
        FROM reddit_posts_enhanced
        WHERE COALESCE(NULLIF(title_cleaned,''), title) IS NOT NULL
    """, conn)
    conn.close()
    if df.empty:
        raise RuntimeError("no data")
    tagged = [TaggedDocument(tok(t), [i]) for i,t in enumerate(df["message"].tolist())]
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = np.vstack([model.infer_vector(tok(t)) for t in df["message"]])
    k = max(1, min(n_clusters, len(df)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)
    labels = kmeans.labels_
    pca = PCA(n_components=2, random_state=42).fit(vectors)
    xy = pca.transform(vectors)
    mdf = pd.DataFrame({"post_id": df["post_id"], "message": df["message"], "cluster": labels, "x": xy[:,0], "y": xy[:,1]})
    kw = {}
    for c in sorted(mdf["cluster"].unique()):
        texts = mdf.loc[mdf["cluster"]==c,"message"].tolist()
        if not texts:
            kw[c]=[]
            continue
        vec = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()
        avg = np.asarray(X.mean(axis=0)).ravel()
        top = np.argsort(-avg)[:10]
        kw[c] = [terms[i] for i in top]
    mdf["cluster_keywords"] = mdf["cluster"].map(kw)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    model.save(str(Path(outdir,"doc2vec.model")))
    with open(Path(outdir,"kmeans.pkl"),"wb") as f: pickle.dump(kmeans,f)
    with open(Path(outdir,"pca.pkl"),"wb") as f: pickle.dump(pca,f)
    mdf.to_csv(Path(outdir,"dataset.csv"), index=False)
    np.save(Path(outdir,"vectors.npy"), vectors)
    fig = px.scatter(mdf, x="x", y="y", color="cluster", hover_name="message", hover_data={"cluster":True,"cluster_keywords":True,"x":False,"y":False}, title="Reddit Posts Clusters")
    fig.write_html(str(Path(outdir,"clusters_latest.html")))
    conn = mysql.connector.connect(**db_config)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reddit_post_clusters (
        post_id VARCHAR(255) PRIMARY KEY,
        cluster_id INT,
        dist_to_centroid DOUBLE,
        x DOUBLE,
        y DOUBLE,
        trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    rows = []
    for i,row in mdf.iterrows():
        cid = int(row["cluster"])
        d = float(np.sum((vectors[i] - kmeans.cluster_centers_[cid])**2))
        rows.append((row["post_id"], cid, d, float(row["x"]), float(row["y"])))
    cur.executemany("""
    INSERT INTO reddit_post_clusters (post_id, cluster_id, dist_to_centroid, x, y)
    VALUES (%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
      cluster_id=VALUES(cluster_id),
      dist_to_centroid=VALUES(dist_to_centroid),
      x=VALUES(x), y=VALUES(y),
      trained_at=CURRENT_TIMESTAMP
    """, rows)
    conn.commit()
    cur.close()
    conn.close()
    return model, kmeans, pca, mdf, vectors

def load_artifacts(outdir="artifacts"):
    model = Doc2Vec.load(str(Path(outdir,"doc2vec.model")))
    with open(Path(outdir,"kmeans.pkl"),"rb") as f: kmeans = pickle.load(f)
    with open(Path(outdir,"pca.pkl"),"rb") as f: pca = pickle.load(f)
    df = pd.read_csv(Path(outdir,"dataset.csv"))
    vectors = np.load(Path(outdir,"vectors.npy"))
    return model, kmeans, pca, df, vectors

def infer_with_artifacts(model, kmeans, pca, df, vectors, text, top_k=10, outdir="artifacts"):
    v = model.infer_vector(tok(text))
    vv = np.asarray(v, dtype=np.float64).reshape(1, -1)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    cid = int(np.argmin(((centers - vv) ** 2).sum(axis=1)))
    idx = np.where(df["cluster"].to_numpy() == cid)[0]
    if idx.size == 0:
        return cid, df.iloc[[]], None
    vecs = vectors.astype(np.float64, copy=False)
    center = centers[cid]
    d = ((vecs[idx] - center) ** 2).sum(axis=1)
    order = np.argsort(d)[:top_k]
    hits = df.iloc[idx[order]].copy()
    sub = df[df["cluster"] == cid]
    qx, qy = pca.transform(vv)[0]
    fig = px.scatter(sub, x="x", y="y", hover_name="message", title=f"Cluster {cid}")
    fig.add_scatter(x=[float(qx)], y=[float(qy)], mode="markers+text", text=["<query>"], name="query")
    out = str(Path(outdir, f"cluster_{cid}_focused.html"))
    fig.write_html(out)
    return cid, hits, out