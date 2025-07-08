import pandas as pd

# Convertir ratings.dat
ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(
    './ratings.dat',
    sep='::',
    engine='python',
    names=ratings_cols
)
ratings.to_csv('ratings.csv', index=False)
print("ratings.csv creado")

# Convertir movies.dat
movies_cols = ['movieId', 'title', 'genres']
movies = pd.read_csv(
    './movies.dat',
    sep='::',
    engine='python',
    names=movies_cols,
    encoding='latin-1'
)
movies.to_csv('movies.csv', index=False)
print("movies.csv creado")