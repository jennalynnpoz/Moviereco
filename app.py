import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re



@st.cache_data
def load_data():
    movies = pd.read_csv("movie.csv")
    ratings = pd.read_csv("rating.csv")
    movie_data = ratings.head(5000)
    rated_movie_ids = movie_data["movieId"].unique()
    movies = movies[movies["movieId"].isin(rated_movie_ids)].reset_index(drop=True)
    return movies, ratings, movie_data

movies, ratings, small_data = load_data()

def clean_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)  
    return title.strip()

movies["clean_title"] = movies["title"].apply(clean_title)

movies["tags"] = movies["clean_title"] + " " + movies["genres"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])

content_sim = cosine_similarity(tfidf_matrix)

user_movie_matrix = small_data.pivot_table(index="userId", columns="movieId", values="rating")

user_movie_matrix = user_movie_matrix.fillna(0)
cs_sim = cosine_similarity(user_movie_matrix.T)

cOntent = 0.7
cOllab = 0.3

hybrid_sim = (cOntent * content_sim) + (cOllab * cs_sim)

def restore_title(t):
    if ", The" in t:
        return "The " + t.replace(", The", "")
    if ", A" in t:
        return "A " + t.replace(", A", "")
    if ", An" in t:
        return "An " + t.replace(", An", "")
    return t

def recommend(movie_title, n=5):
    try:
        idx = movies.index[movies["clean_title"].str.lower() == movie_title.lower()][0]
    except:
        return ["Movie not found"]
    scores = list(enumerate(hybrid_sim[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [s[0] for s in sorted_scores]
    return movies["clean_title"].iloc[movie_indices].apply(restore_title).tolist()


st.title("Movie Recommender!")
st.write("A Hybrid model using **Content + Collaborative Filtering**")

movie_list = sorted(movies["clean_title"].apply(restore_title).unique())

choice = st.selectbox("Choose or Type a movie:", movie_list)

recs = recommend(choice)
if st.button("Get Recommendations"):
    st.subheader("Recommended Movies:")
for r in recs:
    st.write("•", r)
