# 1. IMPORTS

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 2. LOAD PICKLE FILES

# Load Movie-Based CF model
movie_pivot = pd.read_pickle("movie_pivot.pkl")
knn_model = pd.read_pickle("knn_model.pkl")

# Load User-Based CF model components
train_pivot = pd.read_pickle("train_pivot.pkl")
user_similarity = np.load("user_similarity.npy", allow_pickle=True)
user_index_to_id = joblib.load("user_index_to_id.pkl")
user_id_to_index = joblib.load("user_id_to_index.pkl")

# 3. INIT FLASK
app = Flask(__name__)

# 4. FUNCTION DEFINITIONS

# Movie-based Recommendation
def recommend_movie(movie_name):
    if movie_name not in movie_pivot.index:
        return []

    movie_id = np.where(movie_pivot.index == movie_name)[0][0]
    distance, suggestion = knn_model.kneighbors(
        movie_pivot.iloc[movie_id, :].values.reshape(1, -1), n_neighbors=6
    )

    recommended_titles = [movie_pivot.index[idx] for idx in suggestion[0][1:]]
    return recommended_titles

# User-based Recommendation
def predict_ratings_user(user_id, top_n=5):
    if user_id not in user_id_to_index or user_id not in train_pivot.index:
        return []

    user_idx = user_id_to_index[user_id]
    sim_scores = user_similarity[user_idx]
    top_users_idx = np.argsort(sim_scores)[::-1][:top_n]

    index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
    top_users_ids = [
        index_to_user_id[idx]
        for idx in top_users_idx
        if index_to_user_id[idx] in train_pivot.index
    ]

    top_users_ratings = train_pivot.loc[top_users_ids]
    top_sim_scores = sim_scores[top_users_idx]
    weighted_ratings = np.dot(
        top_sim_scores[:len(top_users_ratings)], top_users_ratings
    )
    sum_weights = np.sum(top_sim_scores[:len(top_users_ratings)]) or 1
    pred_ratings = weighted_ratings / sum_weights

    pred_ratings_series = pd.Series(pred_ratings, index=train_pivot.columns)
    user_rated = train_pivot.loc[user_id]
    unrated_mask = user_rated == 0
    pred_ratings_series = pred_ratings_series[unrated_mask]

    return list(pred_ratings_series.sort_values(ascending=False).head(top_n).index)

# 5. API ROUTES

@app.route("/")
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route("/movie_recommend", methods=["GET"])
def movie_recommendation():
    movie_name = request.args.get("movie")

    if not movie_name:
        return jsonify({"error": "Please provide a movie name using ?movie="}), 400

    recommendations = recommend_movie(movie_name)

    if not recommendations:
        return jsonify({"message": "Movie not found or no recommendations."}), 404

    return jsonify({"movie": movie_name, "recommendations": recommendations})

@app.route("/user_recommend", methods=["GET"])
def user_recommendation():
    user_id = request.args.get("user_id")
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "Invalid user_id"}), 400

    recommendations = predict_ratings_user(user_id, top_n=10)

    if not recommendations:
        return jsonify({"message": f"No recommendations for user {user_id}"}), 404

    return jsonify({"user_id": user_id, "recommendations": recommendations})


@app.route("/combined_recommend", methods=["GET"])
def combined_recommendation():
    movie_name = request.args.get("movie")
    user_id = request.args.get("user_id")

    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "Invalid or missing user_id"}), 400

    if not movie_name:
        return jsonify({"error": "Missing movie name (?movie=...)"}), 400

    # Get recommendations
    movie_recs = recommend_movie(movie_name)
    user_recs = predict_ratings_user(user_id, top_n=10)

    # Build response
    return jsonify({
        "movie_input": movie_name,
        "movie_recommendations": movie_recs if movie_recs else "No movie recommendations found.",
        "user_id": user_id,
        "user_recommendations": user_recs if user_recs else "No user recommendations found."
    })

# 6. RUN APP
# if __name__ == "__main__":
   # app.run(debug=True, port=8000)

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))




# import requests
# response = requests.get("http://127.0.0.1:5000/user_recommend", params={"user_id": 1})
# print(response.json())

# code to run in browswer

# Access both recommendations:
#  http://127.0.0.1:8000/combined_recommend?movie&user_id

# Movie-based CF:
# http://127.0.0.1:5000/movie_recommend?movie
# http://127.0.0.1:5000/movie_recommend?movie=Toy Story (1995)

# User-based CF:
# http://127.0.0.1:5000/user_recommend?user_id
