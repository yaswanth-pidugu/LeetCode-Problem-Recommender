import pandas as pd
import numpy as np
import re, ast, pickle
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from pathlib import Path

def clean_title(text):
    text = str(text).lower().strip()
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def to_tag_list(s):
    if isinstance(s, str) and s.startswith('['):
        try:
            vals = ast.literal_eval(s)
            return [str(v).lower().strip() for v in vals]
        except Exception:
            pass
    return [t.strip().lower() for t in str(s).split(',') if t.strip()]

def minmax(x):
    x = x.fillna(0)
    rng = x.max() - x.min()
    return (x - x.min())/rng if rng != 0 else x*0

def load_resources(
    data_path=r"C:\Users\yashw\PycharmProjects\PythonProject4\Data_Pipeline\preprocessed_data.csv",
    model_path=r"C:\Users\yashw\PycharmProjects\PythonProject4\models\lightgbm_model.pkl",
    embed_cache=r"C:\Users\yashw\PycharmProjects\PythonProject4\sbert_embeddings.pkl"
):

    # Load data
    df = pd.read_csv(data_path)
    df["clean_title"] = df["title"].apply(clean_title)
    df["tag_list"] = df["topic_tags"].apply(to_tag_list)
    df["difficulty"] = df["difficulty"].fillna("Medium")

    # Load embeddings
    cache_path = Path(embed_cache)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        embeddings = np.array(cache["embeddings"], dtype=np.float32)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            df["title"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"model_name": "all-MiniLM-L6-v2", "embeddings": embeddings}, f)

    # Tag similarity (Jaccard)
    N = len(df)
    tag_sims = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        tags_i = set(df.iloc[i]["tag_list"])
        for j in range(i+1, N):
            tags_j = set(df.iloc[j]["tag_list"])
            if tags_i and tags_j:
                inter = len(tags_i & tags_j)
                uni = len(tags_i | tags_j)
                if uni:
                    val = inter / uni
                    tag_sims[i,j] = tag_sims[j,i] = val

    # Difficulty similarity
    ladder = {"easy":0, "medium":1, "hard":2}
    diff_vals = df["difficulty"].str.lower().map(ladder).fillna(1).to_numpy(dtype=np.int8)
    diff_sims = np.ones((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            d = abs(diff_vals[i]-diff_vals[j])
            s = 1.0 if d==0 else 0.7 if d==1 else 0.4
            diff_sims[i,j] = diff_sims[j,i] = s

    # Popularity normalization
    acc = minmax(df["acceptance"]) if "acceptance" in df else 0
    likes = minmax(df["likes"]) if "likes" in df else 0
    subs  = minmax(df["submission"]) if "submission" in df else 0
    popularity_score = (0.3*acc + 0.5*likes + 0.2*subs).fillna(0).to_numpy(dtype=np.float32)

    # Load LightGBM model
    model = pickle.load(open(model_path, "rb"))

    return df, embeddings, tag_sims, diff_sims, popularity_score, model


def get_recommendations(idx, df, embeddings, tag_sims, diff_sims, popularity_score, model, k=10):
    emb_sims = np.dot(embeddings[idx], embeddings.T)
    tag_vals = tag_sims[idx]
    diff_vals = diff_sims[idx]
    pop_diffs = np.abs(popularity_score[idx] - popularity_score)
    feats = np.stack([emb_sims, tag_vals, diff_vals, pop_diffs], axis=1)
    scores = model.predict(feats)
    scores[idx] = -1e9  # exclude self
    top_idx = np.argsort(scores)[-k:][::-1]
    return df.iloc[top_idx][["title", "difficulty", "topic_tags"]]

if __name__ == "__main__":
    df, emb, tag, diff, pop, model = load_resources()
    recs = get_recommendations(10, df, emb, tag, diff, pop, model)
    print("\nTop Recommendations:\n", recs)