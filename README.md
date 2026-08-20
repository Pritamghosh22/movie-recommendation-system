# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python, TF-IDF, cosine similarity, and Streamlit.

The application recommends movies similar to a selected movie based on features such as genres, keywords, cast, director, and overview.

## Features

- Search and select a movie
- Get 3 to 10 movie recommendations
- View movie genres
- View director information
- View movie overview
- Similarity score for each recommendation
- Interactive Streamlit web interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Cosine Similarity

## Dataset

The project uses the TMDB 5000 Movie Dataset:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## How It Works

1. Movie and credits datasets are loaded and merged.
2. Relevant features are extracted:
   - Genres
   - Keywords
   - Cast
   - Director
   - Overview
3. These features are combined into a single text representation.
4. TF-IDF converts the text into numerical vectors.
5. Cosine similarity calculates similarity between movies.
6. Movies with the highest similarity scores are recommended.

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt