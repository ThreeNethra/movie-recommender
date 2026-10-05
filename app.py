import json
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_names(text):
    # genres and keywords are stored like [{"id": 28, "name": "Action"}, ...]
    try:
        return [i["name"] for i in json.loads(text)]
    except (TypeError, ValueError):
        return []


def join_names(names):
    # "Science Fiction" -> "ScienceFiction" so it stays one word
    return " ".join(n.replace(" ", "") for n in names)


def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


@st.cache_resource
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[["title", "overview", "genres", "keywords", "tagline"]]
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    df = df.fillna("")

    df["genres"] = df["genres"].apply(get_names)
    df["keywords"] = df["keywords"].apply(get_names)
    df["genre_text"] = df["genres"].apply(", ".join)
    df["genres"] = df["genres"].apply(join_names)
    df["keywords"] = df["keywords"].apply(join_names)
    df["tags"] = df["overview"] + " " + df["tagline"] + " " + df["genres"] + " " + df["keywords"]
    df["clean_text"] = df["tags"].apply(clean)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df["clean_text"])
    return df, matrix


def recommend(item_name, top_n=5):
    idx = movies[movies["title"] == item_name].index[0]
    # only this movie's row is compared with the rest, the full 4800x4800
    # matrix takes too much memory on the free Render instance
    scores = cosine_similarity(matrix[idx], matrix).flatten()
    order = scores.argsort()[::-1]
    result = []
    for i in order:
        if i == idx:
            continue
        result.append((movies.loc[i, "title"], movies.loc[i, "genre_text"], scores[i]))
        if len(result) == top_n:
            break
    return result


st.title("Movie Recommendation System")
st.write("Select a movie and get similar movies based on its overview, genres and keywords.")

movies, matrix = load_data()

selected = st.selectbox("Choose a movie", sorted(movies["title"]))
n = st.slider("How many recommendations?", 3, 10, 5)

if st.button("Recommend"):
    recs = recommend(selected, n)
    st.subheader(f"Movies similar to {selected}")
    rows = []
    for rank, (title, genres, score) in enumerate(recs, start=1):
        rows.append({"#": rank, "Movie": title, "Genres": genres, "Similarity": round(float(score), 3)})
    st.table(pd.DataFrame(rows).set_index("#"))
