import os
import sys
import zipfile
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
from jinja2 import Template
# ---------- Configuration ----------
ML_ZIP = "ml-32m.zip"
ML_DIR = "ml-32m"
RATINGS_IN_DIR = os.path.join(ML_DIR, "ratings.csv")
MOVIES_IN_DIR = os.path.join(ML_DIR, "movies.csv")
OUT_RATINGS = "ratings.csv"
OUT_MOVIES = "movies_with_moods.csv"
MODEL_FILE = "svd_model.npz"
META_FILE = MODEL_FILE + ".meta.pkl"
# Heuristic mapping genre -> moods
GENRE_TO_MOODS = {
    "Comedy": ["happy","lighthearted"],
    "Romance": ["romantic","feelgood"],
    "Drama": ["dramatic","serious"],
    "Action": ["excited","adrenaline"],
    "Horror": ["scary","tense"],
    "Thriller": ["tense","suspenseful"],
    "Sci-Fi": ["curious","wonder"],
    "Documentary": ["informative","thoughtful"],
    "Animation": ["family","lighthearted"],
    "Fantasy": ["wonder","escapist"],
    "Adventure": ["excited","adventure"],
    "Mystery": ["suspenseful","tense"],
}
# ---------- Prepare step ----------
def ensure_extracted():
    if not os.path.isdir(ML_DIR):
        if os.path.exists(ML_ZIP):
            print(f"Extracting {ML_ZIP} ...")
            with zipfile.ZipFile(ML_ZIP, 'r') as z:
                z.extractall()
            if not os.path.isdir(ML_DIR):
                raise FileNotFoundError(f"Extraction didn't produce {ML_DIR}/ — check zip contents.")
        else:
            raise FileNotFoundError(f"Neither {ML_DIR} directory nor {ML_ZIP} found. Put ml-32m.zip here or extracted ml-32m/ directory.")
    if not os.path.exists(RATINGS_IN_DIR) or not os.path.exists(MOVIES_IN_DIR):
        raise FileNotFoundError("Expected ratings.csv and movies.csv inside ml-32m/ after extraction.")
def build_movies_with_moods(movies_csv_path=MOVIES_IN_DIR, out_path=OUT_MOVIES):
    print("Loading movies.csv from", movies_csv_path)
    movies = pd.read_csv(movies_csv_path)
    def infer_moods(genres):
        if pd.isna(genres) or genres.strip() == "":
            return ""
        moods = set()
        for g in genres.split("|"):
            if g in GENRE_TO_MOODS:
                moods.update(GENRE_TO_MOODS[g])
        return "|".join(sorted(moods))
    movies['moods'] = movies['genres'].apply(infer_moods)
    movies.to_csv(out_path, index=False)
    print("Wrote movies_with_moods.csv ->", out_path)
    return out_path
def create_ratings_sample(in_path=RATINGS_IN_DIR, out_path=OUT_RATINGS, max_rows=5_000_000):
    print("Loading ratings.csv (may be large) ...")
    df = pd.read_csv(in_path)
    total = len(df)
    print(f"Found {total:,} ratings.")
    if total > max_rows:
        print(f"Sampling {max_rows:,} ratings for prototyping (change max_rows in script if you want more).")
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print("Wrote ratings.csv ->", out_path)
    return out_path
def prepare_pipeline():
    ensure_extracted()
    build_movies_with_moods()
    create_ratings_sample()
# ---------- Training step ----------
def build_sparse_matrix(ratings_csv):
    df = pd.read_csv(ratings_csv)
    if not set(['userId','movieId','rating']).issubset(df.columns):
        raise ValueError("ratings.csv must contain userId,movieId,rating columns")
    users = np.sort(df['userId'].unique())
    movies = np.sort(df['movieId'].unique())
    user_map = {int(u): i for i, u in enumerate(users)}
    movie_map = {int(m): j for j, m in enumerate(movies)}
    n_users = len(users); n_movies = len(movies)
    print(f"Users: {n_users:,}, Movies: {n_movies:,}, Ratings: {len(df):,}")
    rows = df['userId'].map(user_map).astype(np.int32)
    cols = df['movieId'].map(movie_map).astype(np.int32)
    data = df['rating'].astype(np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
    return R, users, movies, user_map, movie_map
def center_matrix(R_csr):
    print("Centering matrix by subtracting per-user mean ...")
    user_sum = np.array(R_csr.sum(axis=1)).ravel()
    user_count = np.diff(R_csr.indptr)
    user_mean = np.zeros_like(user_sum, dtype=np.float64)
    nonzero_mask = user_count > 0
    user_mean[nonzero_mask] = user_sum[nonzero_mask] / user_count[nonzero_mask]
    R_coo = R_csr.tocoo()
    centered_data = R_coo.data - user_mean[R_coo.row]
    R_centered = sp.csr_matrix((centered_data, (R_coo.row, R_coo.col)), shape=R_csr.shape)
    return R_centered, user_mean
def train_svd(R_centered, k):
    print(f"Computing truncated SVD with k={k} ... (this can take time)")
    t0 = time.time()
    U, s, Vt = svds(R_centered, k=k, return_singular_vectors=True)
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]
    t1 = time.time()
    print(f"SVD done in {t1-t0:.1f}s")
    return U, s, Vt
def save_model(out_file, U, s, Vt, user_mean, users, movies, user_map, movie_map):
    np.savez_compressed(out_file, U=U, s=s, Vt=Vt)
    meta = {
        'user_mean': user_mean,
        'users': users.tolist(),
        'movies': movies.tolist(),
        'user_map': user_map,
        'movie_map': movie_map
    }
    with open(out_file + ".meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Saved model:", out_file)
    print("Saved metadata:", out_file + ".meta.pkl")
def train_pipeline(k=80, ratings_csv=OUT_RATINGS):
    R, users, movies, user_map, movie_map = build_sparse_matrix(ratings_csv)
    R_centered, user_mean = center_matrix(R)
    U, s, Vt = train_svd(R_centered, k=k)
    save_model(MODEL_FILE, U, s, Vt, user_mean, users, movies, user_map, movie_map)
# ---------- Server (FastAPI) ----------
app = FastAPI(title="Single-file Mood SVD Recommender")
HTML_TEMPLATE = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mood SVD Recommender</title>
  <style>
    body { font-family: sans-serif; padding: 24px; max-width: 900px; margin: auto; }
    form { margin-bottom: 24px; }
    input, select { padding: 6px; font-size: 14px; }
    .movie { margin: 8px 0; padding: 8px; border-bottom: 1px solid #eee; }
    .title { font-weight: 600; }
    .meta { color: #555; font-size: 13px; }
  </style>
</head>
<body>
  <h1>Mood-based Movie Recommender (SVD)</h1>
  <p>Enter a userId from the dataset (e.g., 1 to 162541) or leave blank for generic recommendations. Then pick a mood to filter.</p>
  <form method="post" action="/recommend">
    <label>UserId: <input name="user_id" value="{{ user_id or '' }}" /></label>
    &nbsp;&nbsp;
    <label>Mood:
      <select name="mood">
        <option value="">(any mood)</option>
        {% for m in moods %}
        <option value="{{m}}" {% if m==selected_mood %}selected{% endif %}>{{m}}</option>
        {% endfor %}
      </select>
    </label>
    &nbsp;&nbsp;
    <label>Top N: <input name="top_n" value="{{ top_n }}" style="width:60px" /></label>
    &nbsp;&nbsp;
    <button type="submit">Get recommendations</button>
  </form>
  {% if recommendations is not none %}
    <h2>Recommendations for '{% if user_id %}{{user_id}}{% else %}Guest{% endif %}' {% if selected_mood %}in mood '{{selected_mood}}'{% endif %}</h2>
    {% if recommendations|length == 0 %}
      <p>No recommendations found. Try a different user, mood, or remove the mood filter.</p>
    {% else %}
      {% for r in recommendations %}
        <div class="movie">
          <div class="title">{{ r.title }}</div>
          <div class="meta">movieId: {{ r.movieId }} — score: {{ "%.2f"|format(r.score) }} — moods: {{ r.moods }}</div>
        </div>
      {% endfor %}
    {% endif %}
  {% endif %}
  <hr/>
  <p>Model status: {{ model_status }}</p>
</body>
</html>
""")
def load_model_if_exists():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
        return None
    try:
        data = np.load(MODEL_FILE, allow_pickle=True)
        U, s, Vt = data['U'], data['s'], data['Vt']
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        return U, s, Vt, meta
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
def _get_all_moods():
    moods = []
    if os.path.exists(OUT_MOVIES):
        df = pd.read_csv(OUT_MOVIES)
        if 'moods' in df.columns:
            vals = df['moods'].dropna().unique()
            moods_set = set()
            for v in vals:
                if isinstance(v, str) and v.strip():
                    for part in v.split("|"):
                        moods_set.add(part.strip())
            moods = sorted(moods_set)
    return moods
@app.get("/", response_class=HTMLResponse)
async def index():
    model = load_model_if_exists()
    model_status = "model present" if model else "model NOT found - run this script to train"
    moods = _get_all_moods()
    return HTML_TEMPLATE.render(recommendations=None, moods=moods, selected_mood=None, top_n=10, user_id='', model_status=model_status)
@app.post("/recommend", response_class=HTMLResponse)
async def recommend_form(request: Request, user_id: str = Form(''), mood: str = Form(''), top_n: int = Form(10)):
    model = load_model_if_exists()
    model_status = "model present" if model else "model NOT found - run this script to train"
    moods = _get_all_moods()
    if not model:
        return HTML_TEMPLATE.render(recommendations=[], moods=moods, selected_mood=mood, top_n=top_n, user_id=user_id, model_status=model_status)
    U, s, Vt, meta = model
    users, movies, user_map = meta['users'], meta['movies'], meta['user_map']
    user_mean = np.array(meta['user_mean'])
    try:
        uid = int(user_id) if user_id.strip() != "" else None
    except ValueError:
        uid = None
    if uid is not None and uid in user_map:
        idx_u = user_map[uid]
        scores_centered = (U[idx_u, :] * s) @ Vt
        scores = scores_centered + user_mean[idx_u]
    else:
        mean_u_vec = U.mean(axis=0)
        scores_centered = (mean_u_vec * s) @ Vt
        scores = scores_centered + user_mean.mean()
    top_candidate_indices = np.argsort(-scores)[:1000]
    movies_df = pd.read_csv(OUT_MOVIES) if os.path.exists(OUT_MOVIES) else pd.DataFrame(columns=['movieId','title','moods'])
    movies_df.set_index('movieId', inplace=True)
    mood_lower = mood.lower().strip() if mood else None
    recommendations = []
    for idx in top_candidate_indices:
        movie_id = int(movies[idx])
        try:
            row = movies_df.loc[movie_id]
            title = row['title']
            moods_txt = row['moods'] if 'moods' in row and pd.notna(row['moods']) else ""
        except KeyError:
            title = f"Unknown Movie (ID: {movie_id})"
            moods_txt = ""
        entry = {'movieId': movie_id, 'title': title, 'score': float(scores[idx]), 'moods': moods_txt}
        if mood_lower:
            if mood_lower in str(moods_txt).lower():
                recommendations.append(entry)
        else:
            recommendations.append(entry)
        if len(recommendations) >= int(top_n):
            break
    return HTML_TEMPLATE.render(recommendations=recommendations, moods=moods, selected_mood=mood, top_n=top_n, user_id=user_id, model_status=model_status)
# ---------- Smart Runner ----------
if __name__ == "__main__":
    # Check if the full model is already trained
    if os.path.exists(MODEL_FILE):
        print("✅ Model already trained. Starting the web server...")
    else:
        # Check if data is already prepared
        if os.path.exists(OUT_RATINGS) and os.path.exists(OUT_MOVIES):
            print("✅ Data is already prepared. Starting model training...")
        else:
            print("▶️ Step 1: Preparing data...")
            prepare_pipeline()
            print("✅ Data preparation finished.")
        print("\n▶️ Step 2: Training the model (this might take a few minutes)...")
        train_pipeline(k=80)
        print("✅ Model training finished.")
    # --- Always run the server as the final step ---
    print("\n▶️ Step 3: Starting the web server on http://127.0.0.1:8000")
    print("Open your web browser to that address to use the app.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
