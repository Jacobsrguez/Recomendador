import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, SVD, BaselineOnly
from surprise.model_selection import train_test_split
from surprise import accuracy

# Cargar datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# INFO DEL DATASET
st.sidebar.title("📊 Información del Dataset")

# Número de usuarios únicos
num_users = ratings['userId'].nunique()
# Número de películas únicas
num_movies = ratings['movieId'].nunique()
# Número total de valoraciones
num_ratings = ratings.shape[0]
# Número de categorías únicas (extraídas del campo 'genres')
all_genres = movies['genres'].str.split('|').explode().unique()
num_genres = len(all_genres)

st.sidebar.markdown(f"👥 **Usuarios únicos:** {num_users}")
st.sidebar.markdown(f"🎞️ **Películas distintas:** {num_movies}")
st.sidebar.markdown(f"⭐ **Valoraciones totales:** {num_ratings}")
st.sidebar.markdown(f"🏷️ **Categorías únicas:** {num_genres}")
st.sidebar.markdown("📚 **Categorías:**")
st.sidebar.write(", ".join(sorted(all_genres)))

# Selección de usuario para ver cuántas valoraciones ha hecho
selected_user_info = st.sidebar.selectbox("🔍 Ver valoraciones de un usuario", sorted(ratings['userId'].unique()))
user_ratings_count = ratings[ratings['userId'] == selected_user_info].shape[0]
st.sidebar.markdown(f"📝 **Valoraciones del usuario {selected_user_info}:** {user_ratings_count}")

# Conversion de el dataset para libreria Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


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

# Peliculas que no han sido vistas por un usuario
def get_unseen_movies(user_id, ratings_df):
    seen_movies = ratings_df[ratings_df.userId == user_id]['movieId'].tolist()
    all_movies = ratings_df['movieId'].unique()
    return [movie for movie in all_movies if movie not in seen_movies]

# Recomendación de películas
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
st.title("🎬 Recomendador de Películas")

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
    st.subheader("📈 Comparación de modelos")
    st.dataframe(eval_df)

    # Gráfico de barras mejorado
    st.subheader("🔬 RMSE por modelo")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(eval_df["Model"], eval_df["RMSE"], width=0.5)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Comparación de RMSE entre modelos", fontsize=14)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    # Mostrar valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha='center', fontsize=10)

    st.pyplot(fig)

# Análisis de distribución
st.markdown("## 📈 Distribución de Valoraciones")

# Histograma de valoraciones por usuario
st.subheader("👥 Valoraciones por Usuario")
st.caption("Distribución de valoraciones por usuario (usuarios con < 1000 valoraciones)")

user_rating_counts = ratings.groupby('userId').size()
filtered_user_ratings = user_rating_counts[user_rating_counts < 1000]

fig, ax = plt.subplots()
ax.hist(filtered_user_ratings, bins=50, color='skyblue', edgecolor='black')
ax.set_xlabel("Número de valoraciones")
ax.set_ylabel("Cantidad de usuarios")
st.pyplot(fig)

# Histograma de valoraciones por película
st.subheader("🎞️ Valoraciones por Película")
st.caption("Distribución de valoraciones por película (películas con < 100 valoraciones)")

movie_rating_counts = ratings.groupby('movieId').size()
filtered_movie_ratings = movie_rating_counts[movie_rating_counts < 100]

fig, ax = plt.subplots()
ax.hist(filtered_movie_ratings, bins=50, color='lightgreen', edgecolor='black')
ax.set_xlabel("Número de valoraciones")
ax.set_ylabel("Cantidad de películas")
st.pyplot(fig)

st.subheader("👑 Top 5 Usuarios Más Activos")
top_users = ratings['userId'].value_counts().head(5).reset_index()
top_users.columns = ['Usuario', 'Cantidad de Valoraciones']
st.table(top_users)

st.subheader("🍿 Top 5 Películas Más Valoradas")
top_movies = ratings['movieId'].value_counts().head(5).reset_index()
top_movies.columns = ['movieId', 'Cantidad de Valoraciones']
top_movies = top_movies.merge(movies, on='movieId')
st.table(top_movies[['title', 'Cantidad de Valoraciones']])