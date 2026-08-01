# 🎬 Mood-Based Movie Recommender System using SVD

This project is a movie recommendation system built using
Singular Value Decomposition (SVD) and FastAPI.

Features
- Mood based filtering
- Collaborative filtering recommendation
- MovieLens dataset
- Web interface using FastAPI

How to run

1. Install requirements
pip install numpy pandas scipy fastapi uvicorn

2. Run the program
python moodbasedmovie.py

3. Open browser
http://127.0.0.1:8000

A movie recommendation system that combines **Collaborative Filtering** with **Mood-Based Filtering** to provide personalized movie suggestions. The system predicts user preferences using **Singular Value Decomposition (SVD)** and then filters recommendations based on the user's selected mood.

---

## 📖 Overview

Choosing a movie from thousands of available options can be overwhelming. Traditional recommendation systems generally rely on popularity or previous viewing history and often ignore an important factor—the user's current mood.

This project addresses that limitation by combining **matrix factorization** with **mood-aware filtering** to generate more contextually relevant recommendations.

The recommendation pipeline works in two stages:

1. Predict movie ratings using **Singular Value Decomposition (SVD)**.
2. Filter predicted movies according to the mood selected by the user.

---

## ✨ Features

- Mood-based movie recommendations
- Collaborative filtering using Singular Value Decomposition (SVD)
- Personalized recommendations based on predicted ratings
- FastAPI-based web interface
- Genre-to-mood mapping
- Supports filtering by mood
- Uses the MovieLens dataset

---

# Objectives

- Identify the user's current mood.
- Generate personalized movie recommendations.
- Learn hidden user preferences using SVD.
- Predict ratings for unseen movies.
- Improve recommendation quality using mood-based filtering.

---

# System Workflow

```text
User
   │
   ▼
Select Mood
   │
   ▼
MovieLens Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
User-Movie Rating Matrix
   │
   ▼
Matrix Normalization
   │
   ▼
Singular Value Decomposition (SVD)
   │
   ▼
Predicted Ratings
   │
   ▼
Mood Filtering
   │
   ▼
Top Recommended Movies
```

---

# Methodology

### Step 1 – User Input

The user:

- Selects their current mood
- (Optionally) provides preferences

---

### Step 2 – Data Processing

The system:

- Reads MovieLens ratings
- Processes movie genres
- Maps genres to moods
- Builds the user-item rating matrix

---

### Step 3 – Matrix Factorization

The rating matrix is decomposed into

\[
R = U \Sigma V^T
\]

where

- **U** → User latent features
- **Σ** → Importance of latent factors
- **Vᵀ** → Movie latent features

---

### Step 4 – Rating Prediction

Missing ratings are predicted by reconstructing

\[
\hat{R}=U\Sigma V^T
\]

The highest predicted ratings become recommendations.

---

### Step 5 – Mood Filtering

The predicted movies are filtered according to the selected mood.

Example:

| Mood | Example Genres |
|-------|----------------|
| Happy | Comedy |
| Romantic | Romance |
| Excited | Action, Adventure |
| Curious | Sci-Fi |
| Scary | Horror |
| Thoughtful | Documentary |

---

# Mathematical Background

Singular Value Decomposition is a matrix factorization technique that decomposes a sparse rating matrix into three smaller matrices.

\[
R = U\Sigma V^T
\]

where

- **R** = User–Movie Rating Matrix
- **U** = User latent factor matrix
- **Σ** = Singular values
- **Vᵀ** = Movie latent factor matrix

SVD uncovers hidden relationships between users and movies, allowing prediction of ratings for unseen movies.

---

# Data Preprocessing

The system performs:

- Loading MovieLens dataset
- Genre extraction
- Mood generation from genres
- Sparse matrix construction
- User mean normalization
- Missing value estimation
- Matrix factorization

---

# Recommendation Pipeline

```text
MovieLens Dataset
        │
        ▼
Ratings Matrix
        │
        ▼
Normalize Ratings
        │
        ▼
Sparse Matrix
        │
        ▼
SVD Training
        │
        ▼
Predicted Ratings
        │
        ▼
Mood Filter
        │
        ▼
Top-N Recommendations
```

---

# Evaluation Metrics

The recommender can be evaluated using:

- RMSE (Root Mean Square Error)
- Precision
- Recall
- F1-Score

---

# Technologies Used

### Programming Language

- Python

### Framework

- FastAPI

### Libraries

- NumPy
- Pandas
- SciPy
- Jinja2
- Uvicorn

---

# Dataset

MovieLens 32M Dataset

https://grouplens.org/datasets/movielens/

The dataset contains:

- Movie ratings
- Movie titles
- Genres
- User IDs

---

# Installation

Clone the repository

```bash
git clone https://github.com/HarshiniSreeS/mood-movie-recommender.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python moodbasedmovie.py
```

Open

```
http://127.0.0.1:8000
```

---

# Project Structure

```
mood-movie-recommender
│
├── moodbasedmovie.py
├── ratings.csv
├── movies_with_moods.csv
├── svd_model.npz
├── svd_model.npz.meta.pkl
├── README.md
└── requirements.txt
```

---

# Sample Output

- User selects mood
- System predicts movie ratings using SVD
- Movies matching the selected mood are displayed
- Top-N recommendations are shown on the FastAPI webpage

---

# Future Improvements

- Automatic mood detection using NLP
- Emotion detection using facial expressions
- Hybrid recommendation with content-based filtering
- Personalized actor and director preferences
- Real-time user feedback learning
- Deep learning-based recommendation models

---

# Team

**Team 17**
- S Harshini Sree
- Thiyaanesh N R
- Yuvanidhi R
- Prakeya S

Faculty Guide

**Dr. Sandhya S. Pai**

---

# References

- MovieLens Dataset – https://grouplens.org/datasets/movielens/
- FastAPI – https://fastapi.tiangolo.com/
- NumPy – https://numpy.org/
- Pandas – https://pandas.pydata.org/
- SciPy – https://scipy.org/

---

## License

This project is developed for academic purposes.
