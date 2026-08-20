import ast
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge movies and credits datasets
    movies = movies.merge(
        credits,
        on="title"
    )

    # Keep only required columns
    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ]

    # Remove duplicate movie titles
    movies = movies.drop_duplicates(
        subset="title"
    ).reset_index(drop=True)

    # Handle missing overview values
    movies["overview"] = movies["overview"].fillna("")

    return movies


movies = load_data()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_names(text):
    """
    Extract genre and keyword names.
    """

    names = []

    try:

        data = ast.literal_eval(text)

        for item in data:
            names.append(item["name"])

    except (ValueError, SyntaxError, TypeError):
        pass

    return " ".join(names)


def extract_cast(text):
    """
    Extract top 5 cast members.
    """

    names = []

    try:

        data = ast.literal_eval(text)

        for item in data[:5]:
            names.append(item["name"])

    except (ValueError, SyntaxError, TypeError):
        pass

    return " ".join(names)


def extract_director(text):
    """
    Extract director from crew.
    """

    try:

        data = ast.literal_eval(text)

        for item in data:

            if item.get("job") == "Director":
                return item.get("name", "")

    except (ValueError, SyntaxError, TypeError):
        pass

    return ""


# =========================================================
# DATA PREPROCESSING
# =========================================================

@st.cache_data
def preprocess_data(movies):

    movies = movies.copy()

    # Extract genres
    movies["genres"] = movies["genres"].apply(
        extract_names
    )

    # Extract keywords
    movies["keywords"] = movies["keywords"].apply(
        extract_names
    )

    # Extract cast
    movies["cast"] = movies["cast"].apply(
        extract_cast
    )

    # Extract director
    movies["director"] = movies["crew"].apply(
        extract_director
    )

    # Combine important movie information
    movies["combined_features"] = (
        movies["genres"] + " " +
        movies["keywords"] + " " +
        movies["cast"] + " " +
        movies["director"] + " " +
        movies["overview"]
    )

    return movies


movies = preprocess_data(movies)


# =========================================================
# TF-IDF + COSINE SIMILARITY
# =========================================================

@st.cache_resource
def build_similarity_matrix(combined_features):

    # Convert text into numerical vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000
    )

    feature_vectors = vectorizer.fit_transform(
        combined_features
    )

    # Calculate similarity between movies
    similarity_matrix = cosine_similarity(
        feature_vectors
    )

    return similarity_matrix


similarity = build_similarity_matrix(
    movies["combined_features"]
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_movies(movie_name, number_of_movies=5):

    matching_movie = movies[
        movies["title"] == movie_name
    ]

    if matching_movie.empty:
        return []

    movie_index = matching_movie.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(
            similarity[movie_index]
        )
    )

    # Sort by similarity
    sorted_movies = sorted(
        similarity_scores,
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    # First movie is the selected movie itself,
    # therefore we skip it.
    for index, score in sorted_movies[1:number_of_movies + 1]:

        movie = movies.iloc[index]

        recommendations.append(
            {
                "title": movie["title"],
                "genres": movie["genres"],
                "director": movie["director"],
                "overview": movie["overview"],
                "similarity": score
            }
        )

    return recommendations


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 1.15rem;
    opacity: 0.75;
    margin-bottom: 25px;
}

.movie-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.30);
    min-height: 155px;
    margin-bottom: 10px;
}

.movie-title {
    font-size: 1.20rem;
    font-weight: 700;
    margin-bottom: 15px;
}

.movie-meta {
    font-size: 0.90rem;
    opacity: 0.80;
    margin-bottom: 8px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
Discover movies similar to the ones you already love using
content-based recommendation.
</div>
""",
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# MOVIE SELECTION
# =========================================================

movie_list = sorted(
    movies["title"].dropna().unique()
)


selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_list,
    index=None,
    placeholder="Search for a movie..."
)


number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


recommend_button = st.button(
    "✨ Get Recommendations",
    use_container_width=True,
    type="primary"
)


# =========================================================
# DISPLAY RECOMMENDATIONS
# =========================================================

if recommend_button:

    if selected_movie is None:

        st.warning(
            "Please select a movie first."
        )

    else:

        recommendations = recommend_movies(
            selected_movie,
            number_of_recommendations
        )

        if not recommendations:

            st.error(
                "No recommendations could be generated."
            )

        else:

            st.success(
                f"Showing movies similar to {selected_movie}"
            )

            st.subheader(
                "🍿 Recommended Movies"
            )

            # Maximum 5 movies in one row
            for start in range(
                0,
                len(recommendations),
                5
            ):

                current_movies = recommendations[
                    start:start + 5
                ]

                columns = st.columns(
                    len(current_movies)
                )

                for column, movie in zip(
                    columns,
                    current_movies
                ):

                    with column:

                        similarity_percentage = round(
                            movie["similarity"] * 100,
                            1
                        )

                        director = (
                            movie["director"]
                            if movie["director"]
                            else "Unknown"
                        )

                        # Movie card HTML
                        card_html = f"""
<div class="movie-card">
<div class="movie-title">🎞️ {movie["title"]}</div>
<div class="movie-meta"><b>Director:</b> {director}</div>
<div class="movie-meta"><b>Similarity:</b> {similarity_percentage}%</div>
</div>
"""

                        st.markdown(
                            card_html,
                            unsafe_allow_html=True
                        )

                        # Genres
                        if movie["genres"]:

                            st.caption(
                                "🎭 " + movie["genres"]
                            )

                        # Movie overview
                        with st.expander(
                            "Movie details"
                        ):

                            if movie["overview"]:

                                st.write(
                                    movie["overview"]
                                )

                            else:

                                st.write(
                                    "No overview available."
                                )


# =========================================================
# ABOUT SECTION
# =========================================================

st.divider()


with st.expander(
    "ℹ️ About this recommendation system"
):

    st.write(
        """
        This application uses a content-based movie
        recommendation approach.

        Movie information including genres, keywords,
        cast, director and overview is converted into
        numerical feature vectors using TF-IDF.

        Cosine similarity is then calculated between
        movies to identify movies with similar content.

        The movies with the highest similarity scores
        are recommended to the user.
        """
    )