import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD, BaselineOnly
from surprise.model_selection import train_test_split

# Cargar datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Configurar surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Modelos disponibles
def get_model(algorithm_name):
    if algorithm_name == "Item-Item (Cosine)":
        return KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
    elif algorithm_name == "User-User (Cosine)":
        return KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    elif algorithm_name == "SVD":
        return SVD()
    elif algorithm_name == "BaselineOnly":
        return BaselineOnly()
    else:
        return KNNBasic()

# Función para obtener películas no vistas
def get_unseen_movies(user_id, ratings_df):
    seen_movies = ratings_df[ratings_df.userId == user_id]['movieId'].tolist()
    all_movies = ratings_df['movieId'].unique()
    return [movie for movie in all_movies if movie not in seen_movies]

# Función de recomendación
def recommend_movies(user_id, algo, n=10):
    unseen = get_unseen_movies(user_id, ratings)
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    result = pd.DataFrame([{
        "movieId": pred.iid,
        "Predicted Rating": round(pred.est, 2)
    } for pred in top_n])
    result = result.merge(movies, on="movieId", how="left")[['title', 'Predicted Rating']]
    return result

# STREAMLIT INTERFAZ
st.title("🎬 Recomendador de Películas")

user_ids = sorted(ratings['userId'].unique())
selected_user = st.selectbox("👤 Selecciona un usuario", user_ids)

model_options = ["Item-Item (Cosine)", "User-User (Cosine)", "SVD", "BaselineOnly"]
selected_model = st.selectbox("🧠 Selecciona el algoritmo", model_options)

if st.button("🔍 Recomendar"):
    with st.spinner("Entrenando modelo y generando recomendaciones..."):
        algo = get_model(selected_model)
        algo.fit(trainset)
        recommendations = recommend_movies(selected_user, algo)
    st.success(f"Recomendaciones para el usuario {selected_user} usando {selected_model}:")
    st.table(recommendations)
