import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from surprise import Dataset, Reader, KNNBasic, SVD, BaselineOnly, SVDpp, NMF, KNNWithMeans, KNNWithZScore, KNNBaseline, NormalPredictor, SlopeOne, CoClustering
from surprise.model_selection import train_test_split
from surprise import accuracy
from urllib.parse import parse_qs
import numpy as np
import time
from surprise.model_selection import cross_validate


# Cargar datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Verificamos si se solicitó volver al login desde el botón HTML
query_params = st.query_params
if "volver_login" in query_params:
  st.session_state.login_state = "not_logged_in"
  st.session_state.guest_ratings = []
  st.experimental_rerun()

ADMIN_USER = "admin"
ADMIN_PASS = "1234"

def get_model(algorithm_name):
  if algorithm_name == "Item-Item (Cosine)":
    return KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
  elif algorithm_name == "User-User (Cosine)":
    return KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
  elif algorithm_name == "SVD":
    return SVD()
  elif algorithm_name == "SVD++":
    return SVDpp()
  elif algorithm_name == "NMF":
    return NMF()
  elif algorithm_name == "KNNBasic":
    return KNNBasic()
  elif algorithm_name == "KNNWithMeans":
    return KNNWithMeans()
  elif algorithm_name == "KNNWithZScore":
    return KNNWithZScore()
  elif algorithm_name == "KNNBaseline":
    return KNNBaseline()
  elif algorithm_name == "BaselineOnly":
    return BaselineOnly()
  elif algorithm_name == "NormalPredictor":
    return NormalPredictor()
  elif algorithm_name == "SlopeOne":
    return SlopeOne()
  elif algorithm_name == "CoClustering":
    return CoClustering()
  else:
    return KNNBasic()


def r2_score(predictions):
  y_true = np.array([pred.r_ui for pred in predictions])
  y_pred = np.array([pred.est for pred in predictions])
  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
  return 1 - ss_res / ss_tot
  

def precision_recall_at_k(predictions, k=10, threshold=4.0):
  user_est_true = {}
  for pred in predictions:
      user_est_true.setdefault(pred.uid, []).append((pred.est, pred.r_ui))
  
  precisions = []
  recalls = []

  for uid, user_ratings in user_est_true.items():
      user_ratings.sort(key=lambda x: x[0], reverse=True)
      top_k = user_ratings[:k]
      
      n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
      n_rec_k = sum((est >= threshold) for (est, _) in top_k)
      n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)
      
      precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
      recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
      
      precisions.append(precision)
      recalls.append(recall)
  
  return np.mean(precisions), np.mean(recalls)


def f1_at_k(precision, recall):
  if precision + recall == 0:
      return 0
  return 2 * (precision * recall) / (precision + recall)



def mostrar_login():
  st.title("Inicio de Sesión")
  username = st.text_input("👤 Usuario")
  password = st.text_input("🔑 Contraseña", type="password")
  col1, col2 = st.columns([1, 1])
  with col1:
    if st.button("Iniciar sesión"):
      if username == "admin" and password == "1234":
        st.session_state.login_state = "admin"
        st.success("✅ Acceso concedido como administrador.")
        st.rerun()
      else:
        st.error("❌ Usuario o contraseña incorrectos.")

  with col2:
    if st.button("Acceder como invitado"):
      st.session_state.login_state = "guest"
      st.info("Accediendo como invitado...")
      st.rerun()
  st.stop()

def admin_login():
  # INFO DEL DATASET -> Desplegable de la barra lateral
  st.sidebar.title("Información del Dataset")

  # Número de usuarios únicos
  num_users = ratings['userId'].nunique()
  # Número de películas únicas
  num_movies = ratings['movieId'].nunique()
  # Número total de valoraciones
  num_ratings = ratings.shape[0]
  # Número de categorías únicas (extraídas del campo 'genres')
  all_genres = movies['genres'].str.split('|').explode().unique()
  all_genres2 = [genre for genre in all_genres if genre != '(no genres listed)']
  num_genres = len(all_genres2)

  st.sidebar.markdown(f"👥 **Usuarios únicos:** {num_users}")
  st.sidebar.markdown(f"🎞️ **Películas distintas:** {num_movies}")
  st.sidebar.markdown(f"⭐ **Valoraciones totales:** {num_ratings}")
  st.sidebar.markdown(f"🏷️ **Categorías únicas:** {num_genres}")
  st.sidebar.markdown("📚 **Categorías:**")
  st.sidebar.write(", ".join(sorted(all_genres2)))


  # Selección de usuario para ver cuántas valoraciones ha hecho
  selected_user_info = st.sidebar.selectbox("🔍 Ver valoraciones de un usuario", sorted(ratings['userId'].unique()))
  user_ratings_count = ratings[ratings['userId'] == selected_user_info].shape[0]
  st.sidebar.markdown(f"📝 **Valoraciones del usuario {selected_user_info}:** {user_ratings_count}")

  # Selección de un usuario para ver sus generos favoritos
  selected_user_genres = st.sidebar.selectbox("🔍 Ver géneros favoritos de un usuario", sorted(ratings['userId'].unique()))
  user_rated = ratings[ratings['userId'] == selected_user_genres]
  user_genres = user_rated.merge(movies[['movieId', 'genres']], on='movieId')
  user_genres_exploded = user_genres.copy()
  user_genres_exploded['genres'] = user_genres_exploded['genres'].str.split('|')
  user_genres_exploded = user_genres_exploded.explode('genres')
  genre_rating_avg = user_genres_exploded.groupby('genres')['rating'].mean().sort_values(ascending=False)
  top_genres = genre_rating_avg.head(5).index.tolist()
  st.sidebar.markdown(f"🌟 **Géneros favoritos del usuario {selected_user_genres}:** {', '.join(top_genres)}")

  st.sidebar.markdown("---")  # Separador visual
  cerrar = st.sidebar.button("🔴 Cerrar sesión")

  if cerrar:
    st.session_state.login_state = "not_logged_in"
    st.rerun()

  # Conversion de el dataset para libreria Surprise
  reader = Reader(rating_scale=(0.5, 5.0))
  data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
  trainset, testset = train_test_split(data, test_size=0.2, random_state=42)





  # Recomendación de películas
  def recommend_movies(user_id, recommender_model, n=10):
      unseen = get_unseen_movies(user_id, ratings)
      predictions = [recommender_model.predict(user_id, movie_id) for movie_id in unseen]
      predictions.sort(key=lambda x: x.est, reverse=True)
      top_n = predictions[:n]
      result = pd.DataFrame([{
          "movieId": pred.iid,
          "Predicted Rating": round(pred.est, 2)
      } for pred in top_n])
      result = result.merge(movies, on="movieId", how="left")[['title', 'Predicted Rating']]
      return result

  # Evaluar todos los modelos
  def evaluate_models(k_value):
      model_names = [
          "Item-Item (Cosine)", "User-User (Cosine)",
          "SVD", "SVD++", "NMF",
          "KNNBasic", "KNNWithMeans", "KNNWithZScore", "KNNBaseline",
          "BaselineOnly", "NormalPredictor", "SlopeOne", "CoClustering"
      ]
      results = []
      for name in model_names:
          recommender_model = get_model(name)

          # Medir el tiempo de entrenamiento
          start_time = time.time()
          recommender_model.fit(trainset)
          train_time = time.time() - start_time

          # mediel el tiempo de predicción
          start_test = time.time()
          predictions = recommender_model.test(testset)
          test_time = time.time() - start_test

          # Metricas de evaluación
          rmse = accuracy.rmse(predictions, verbose=False)
          mae = accuracy.mae(predictions, verbose=False)
          r2 = r2_score(predictions)
          precision, recall = precision_recall_at_k(predictions, k=k_value, threshold=4.0)
          f1 = f1_at_k(precision, recall)


          results.append({
            "Model": name, 
            "RMSE": rmse, 
            "MAE": mae, 
            "R2": r2,
            "Precision@10": precision,
            "Recall@10": recall,
            "F1@10": f1,
            "Train Time (s)": round(train_time, 2),
            "Test Time (s)": round(test_time, 2)
          })
      return pd.DataFrame(results)
  
  def cross_validate_models():
    models_to_check = {
        "SVD": SVD(),
        "KNNBasic": KNNBasic()
    }

    results = []
    for name, model in models_to_check.items():
        cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        results.append({
            "Model": name,
            "RMSE Mean": round(np.mean(cv_results['test_rmse']), 4),
            "MAE Mean": round(np.mean(cv_results['test_mae']), 4),
            "Fit Time (s)": round(np.mean(cv_results['fit_time']), 2),
            "Test Time (s)": round(np.mean(cv_results['test_time']), 2)
        })
    return pd.DataFrame(results)


  # STREAMLIT INTERFAZ
  st.title("🎬 Recomendador de Películas")

  user_ids = sorted(ratings['userId'].unique())
  selected_user = st.selectbox("👤 Selecciona un usuario", user_ids)

  model_options = [
      "Item-Item (Cosine)", "User-User (Cosine)",
      "SVD", "SVD++", "NMF",
      "KNNBasic", "KNNWithMeans", "KNNWithZScore", "KNNBaseline",
      "BaselineOnly", "NormalPredictor", "SlopeOne", "CoClustering"
  ]

  selected_model = st.selectbox("🧠 Selecciona el algoritmo para recomendar", model_options)

  col1, col2 = st.columns(2)
  with col1:
    recomendar = st.button("🔍 Recomendar películas")
  if recomendar:
    with st.spinner("Entrenando modelo..."):
      recommender_model = get_model(selected_model)
      recommender_model.fit(trainset)
      recommendations = recommend_movies(selected_user, recommender_model)
    st.success(f"🎯 Recomendaciones para el usuario {selected_user} usando {selected_model}:")
    st.table(recommendations)

  with col2:
    k_value = st.sidebar.slider("🎯 Valor de K para Precision/Recall/F1@K", 1, 20, 10)
    evaluar = st.button("📊 Evaluar modelo")
  if st.button("🧪 Validación Cruzada (K-Fold)"):
    with st.spinner("Ejecutando cross-validation..."):
      cv_df = cross_validate_models()
    st.subheader("🧪 Resultados de Cross-Validation (5-Folds)")
    st.dataframe(cv_df)
  if evaluar:
    with st.spinner("Evaluando..."):
      eval_df = evaluate_models(k_value)
    st.subheader("📈 Comparación de modelos")
    st.dataframe(eval_df)

    st.subheader("🔬 Comparación de Modelos - RMSE")

    best_rmse = eval_df["RMSE"].min()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(eval_df["Model"], eval_df["RMSE"], edgecolor="black")

    # Colorear el mejor modelo
    for bar, value in zip(bars, eval_df["RMSE"]):
      if value == best_rmse:
        bar.set_color("gold")
      else:
        bar.set_color("skyblue")

    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por Modelo (más bajo es mejor)")
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.3f}", ha='center', fontsize=9)

    st.pyplot(fig)


    # Gráfico de MAE con color para el mejor modelo
    st.subheader("🏅 Comparación de Modelos - MAE")

    best_mae = eval_df["MAE"].min()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(eval_df["Model"], eval_df["MAE"], edgecolor="black")

    # Colorear el mejor modelo
    for bar, value in zip(bars, eval_df["MAE"]):
      if value == best_mae:
        bar.set_color("limegreen")
      else:
        bar.set_color("lightcoral")

    ax.set_ylabel("MAE")
    ax.set_title("MAE por Modelo (más bajo es mejor)")
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.3f}", ha='center', fontsize=9)

    st.pyplot(fig)

    # Gráfico de Precision@10
    st.subheader("🎯 Comparación de Modelos - Precision@10")
    best_precision = eval_df["Precision@10"].max()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(eval_df["Model"], eval_df["Precision@10"], edgecolor="black")

    for bar, value in zip(bars, eval_df["Precision@10"]):
        if value == best_precision:
            bar.set_color("orange")
        else:
            bar.set_color("lightblue")

    ax.set_ylabel("Precision@10")
    ax.set_title("Precision@10 por Modelo (más alto es mejor)")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.3f}", ha='center', fontsize=9)

    st.pyplot(fig)


    # Gráfico de Recall@10
    st.subheader("🔍 Comparación de Modelos - Recall@10")

    best_recall = eval_df["Recall@10"].max()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(eval_df["Model"], eval_df["Recall@10"], edgecolor="black")

    for bar, value in zip(bars, eval_df["Recall@10"]):
        if value == best_recall:
            bar.set_color("purple")
        else:
            bar.set_color("lightgray")

    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 por Modelo (más alto es mejor)")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.3f}", ha='center', fontsize=9)

    st.pyplot(fig)

    # Grafico de F1@10
    st.subheader("🏅 Comparación de Modelos - F1@10")

    best_f1 = eval_df["F1@10"].max()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(eval_df["Model"], eval_df["F1@10"], edgecolor="black")

    for bar, value in zip(bars, eval_df["F1@10"]):
        if value == best_f1:
            bar.set_color("teal")
        else:
            bar.set_color("salmon")

    ax.set_ylabel("F1@10")
    ax.set_title("F1@10 por Modelo (más alto es mejor)")
    ax.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.3f}", ha='center', fontsize=9)

    st.pyplot(fig)



  # Análisis de distribución
  st.subheader("📈 Gráficas de Valoraciones")
  with st.expander("Mostrar/Ocultar Gráficas de Valoraciones"):
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


def guest():
  # --- Inicializar estado si es necesario
  if "guest_ratings" not in st.session_state:
    st.session_state.guest_ratings = []

  if "current_guest_movie" not in st.session_state:
    top_movies = ratings['movieId'].value_counts().head(200).index.tolist()
    st.session_state.current_guest_movie = random.choice(top_movies)

  # --- Mostrar progreso
  valoradas = len(st.session_state.guest_ratings)
  min_requeridas = 5

  st.markdown(
    """
    <div style="display: flex; justify-content: flex-end;">
      <form action="">
        <button type="submit" style="
          background-color: #ff4b4b;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 8px;
          font-size: 14px;
          cursor: pointer;
        " formaction="?volver_login=1">Volver al login</button>
      </form>
    </div>
    """,
    unsafe_allow_html=True
  )
  st.title("👋 Bienvenido")
  st.markdown(f"🎯 Has valorado **{valoradas} de {min_requeridas}** películas necesarias para obtener recomendaciones.")

  # --- Mostrar película actual
  current_id = st.session_state.current_guest_movie
  movie_title = movies[movies['movieId'] == current_id]['title'].values[0]
  st.subheader(f"🎬 ¿Has visto esta película?")
  st.markdown(f"**{movie_title}**")

  # --- Slider para puntuar
  rating = st.slider("⭐ Valora esta película", 0.5, 5.0, 3.0, step=0.5)

  col1, col2 = st.columns(2)

  with col1:
    if st.button("✅ Valorar"):
      # Guardar valoración
      st.session_state.guest_ratings.append({
        "userId": 999999,
        "movieId": current_id,
        "rating": rating
      })
      # Elegir nueva película
      top_movies = ratings['movieId'].value_counts().head(200).index.tolist()
      ya_vistas = [r["movieId"] for r in st.session_state.guest_ratings]
      posibles = [m for m in top_movies if m not in ya_vistas]
      if posibles:
        st.session_state.current_guest_movie = random.choice(posibles)
      else:
        st.warning("🎉 Has valorado todas las películas de la lista.")

      st.rerun()

  with col2:
    if st.button("🔄 Cambiar película"):
      top_movies = ratings['movieId'].value_counts().head(200).index.tolist()
      ya_vistas = [r["movieId"] for r in st.session_state.guest_ratings]
      posibles = [m for m in top_movies if m != current_id and m not in ya_vistas]
      if posibles:
        st.session_state.current_guest_movie = random.choice(posibles)
        st.rerun()
      else:
        st.warning("⚠️ No hay más películas para mostrar.")

  # --- Si ya valoró el mínimo, mostramos botón para ver recomendaciones
  if valoradas >= min_requeridas:

    if st.button("🎯 Ver recomendaciones"):
      st.session_state.show_recommendations = True

    if st.session_state.show_recommendations:  
      # --- Mostrar recomendaciones si el invitado ya completó las valoraciones ---
      st.success("Listo!. Aquí están tus recomendaciones")
      guest_df = pd.DataFrame(st.session_state.guest_ratings)

      # Combinar valoraciones reales con las del invitado
      combined_df = pd.concat([ratings, guest_df], ignore_index=True)

      reader = Reader(rating_scale=(0.5, 5.0))
      data = Dataset.load_from_df(combined_df[['userId', 'movieId', 'rating']], reader)
      trainset = data.build_full_trainset()

      # Entrenar con SVD++
      recommender_model = SVDpp()
      with st.spinner("Entrenando modelo por favor espera un momento..."):
          recommender_model.fit(trainset)

          def get_unseen_movies_guest():
              seen = guest_df['movieId'].tolist()
              all_movies = ratings['movieId'].unique()
              return [m for m in all_movies if m not in seen]

          def recommend_guest(user_id, recommender_model, n=10):
              unseen = get_unseen_movies_guest()
              predictions = [recommender_model.predict(user_id, movie_id) for movie_id in unseen]
              predictions.sort(key=lambda x: x.est, reverse=True)
              top_n = predictions[:n]
              result = pd.DataFrame([{
                  "movieId": pred.iid,
                  "Predicted Rating": round(pred.est, 2)
              } for pred in top_n])
              result = result.merge(movies, on="movieId", how="left")[['title', 'Predicted Rating']]
              return result

          recs = recommend_guest(999999, recommender_model)
      st.table(recs)
      st.session_state.show_recommendations = False


# Hay que asegurarse de que el estado de sesión está inicializado porque si no existe Streamlit lanza ese AttributeError
if "login_state" not in st.session_state:
    st.session_state.login_state = "not_logged_in"
if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False
if st.session_state.login_state == "not_logged_in":
    mostrar_login()
elif st.session_state.login_state == "admin":
    admin_login()
elif st.session_state.login_state == "guest":
    guest()