# 🎬 Recomendador de Películas

**Autor:** Jacob Santana Rodríguez  
**Correo:** alu0101330426@ull.edu.es  
**Fecha:** 1 de abril de 2025  

---

## 📌 Descripción del Proyecto

Este proyecto consiste en el desarrollo de un **sistema recomendador de películas** basado en técnicas de **filtrado colaborativo**, una metodología comúnmente utilizada en sistemas de recomendación como los de Netflix o Amazon.

La finalidad del proyecto no es solo construir un recomendador, sino también realizar un **análisis comparativo de distintos algoritmos de recomendación**, con el objetivo de evaluar su rendimiento y precisión en contextos reales.

---

## 📚 Dataset Utilizado

Se ha utilizado el **conjunto de datos público [MovieLens](https://grouplens.org/datasets/movielens/)**, que incluye miles de valoraciones hechas por usuarios sobre un extenso catálogo de películas. Este dataset es ampliamente utilizado en investigación y desarrollo de sistemas de recomendación, lo que lo convierte en una base sólida y fiable para este proyecto.

---

## ⚙️ Tecnologías y Algoritmos

- **Surprise** como motor de recomendación
- **Streamlit** como interfaz gráfica
- **Filtrado colaborativo basado en usuarios** (User-User)
- **Filtrado colaborativo basado en ítems** (Item-Item)
- **Modelos de factorización matricial**: SVD, SVD++, NMF
- **Modelos alternativos**: KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, BaselineOnly, SlopeOne, CoClustering
- **Métricas de evaluación**: RMSE, MAE, R², Precision@K, Recall@K, F1@K
- **Validación cruzada (K-Fold)** para comparar el rendimiento de modelos
- **Análisis exploratorio de usuarios y películas**: valoraciones, géneros favoritos, actividad por usuario
- **Modo invitado** con sistema de recomendaciones personalizado

---

## 📝 Licencia

Este proyecto es de carácter académico y de libre uso para fines educativos.

---

## 💻 Instrucciones 

Para ejecutar el codigo, ejecute  
streamlit run filtrado.py


¡Gracias por visitar este proyecto! 🎥✨
