# Movie Recommendation System

Content based movie recommender made for Assignment 20. You pick a movie and it shows the most similar ones based on the plot overview, genres and keywords.

## Dataset

TMDB 5000 Movie Dataset - https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Only `tmdb_5000_movies.csv` is used (4803 movies). `title` is the item identifier and the text comes from `overview`, `tagline`, `genres` and `keywords`.

## Files

- `movie_recommender.ipynb` - notebook with all the steps (loading, preprocessing, TF-IDF, cosine similarity, recommend function and tests)
- `app.py` - Streamlit app
- `requirements.txt`
- `tmdb_5000_movies.csv` - dataset

## Steps

1. Loaded the csv with pandas and checked shape, columns, nulls and duplicates.
2. Kept title, overview, tagline, genres and keywords. Missing values are filled with empty string.
3. genres and keywords are JSON strings, so the names are extracted from them and joined into one `tags` column with overview and tagline.
4. Cleaning: lowercase, remove punctuation/special characters, remove stopwords (scikit-learn's English stopword list). Saved as `clean_text`.
5. `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` on `clean_text`.
6. Cosine similarity between all movies.
7. `recommend(item_name, top_n=5)` finds the index of the movie, sorts its similarity scores and returns the top N.

Cosine similarity is used because it compares the direction of the vectors and ignores their length, so a movie with a long overview and one with a short overview can still match if they are about the same things.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Render as a web service connected to this GitHub repo.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

One change from the notebook: the app does not store the full similarity matrix. It computes the similarity of only the selected movie against the others when the button is clicked. The output is the same but it needs much less memory, which matters on the free instance (512 MB).

Live app: (link will be added after deployment)
