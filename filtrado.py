import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, SVD, BaselineOnly
from surprise.model_selection import train_test_split
from surprise import accuracy

# Cargar datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Configurar surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

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

# Obtener películas no vistas
def get_unseen_movies(user_id, ratings_df):
    seen_movies = ratings_df[ratings_df.userId == user_id]['movieId'].tolist()
    all_movies = ratings_df['movieId'].unique()
    return [movie for movie in all_movies if movie not in seen_movies]

# Recomendar películas
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

# Evaluar todos los modelos
def evaluate_models():
    results = []
    for name in ["Item-Item (Cosine)", "User-User (Cosine)", "SVD", "BaselineOnly"]:
        algo = get_model(name)
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(results)

# STREAMLIT INTERFAZ
st.title("🎬 Recomendador de Películas con Comparación de Modelos")

user_ids = sorted(ratings['userId'].unique())
selected_user = st.selectbox("👤 Selecciona un usuario", user_ids)

model_options = ["Item-Item (Cosine)", "User-User (Cosine)", "SVD", "BaselineOnly"]
selected_model = st.selectbox("🧠 Selecciona el algoritmo para recomendar", model_options)

if st.button("🔍 Recomendar películas"):
    with st.spinner("Entrenando modelo..."):
        algo = get_model(selected_model)
        algo.fit(trainset)
        recommendations = recommend_movies(selected_user, algo)
    st.success(f"🎯 Recomendaciones para el usuario {selected_user} usando {selected_model}:")
    st.table(recommendations)

if st.button("📊 Evaluar todos los modelos"):
    with st.spinner("Evaluando..."):
        eval_df = evaluate_models()
    st.subheader("Comparación de modelos")
    st.dataframe(eval_df)

    # Gráfico de barras
    st.subheader("🔬 RMSE por modelo")
    fig, ax = plt.subplots()
    ax.bar(eval_df["Model"], eval_df["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title("Comparación de RMSE entre modelos")
    st.pyplot(fig)
