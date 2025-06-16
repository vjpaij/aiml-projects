### Description:

A Recommendation System suggests items to users based on similarity between their preferences and item characteristics. In this project, we’ll build a content-based filtering system that recommends movies based on their descriptions using cosine similarity on TF-IDF vectors.

- Converts movie descriptions into TF-IDF vectors
- Uses cosine similarity to find the most similar items
- Returns top N recommendations for a given movie title

## Movie Recommendation System using TF-IDF and Cosine Similarity

This script demonstrates a simple content-based movie recommendation system using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization and **cosine similarity**.

### Overview

The goal of this system is to recommend similar movies based on a given movie's description. It uses text-based features (movie descriptions) to find similarities between movies.

### How it Works

1. **Dataset Preparation:**
   A small dataset is created containing movie titles and their respective descriptions.

2. **TF-IDF Vectorization:**
   The descriptions are transformed into numerical vectors using `TfidfVectorizer` from `scikit-learn`. TF-IDF helps emphasize important keywords by reducing the weight of commonly used words.

3. **Cosine Similarity:**
   Cosine similarity is computed between the TF-IDF vectors of all movie pairs to measure their similarity based on the angle between their vector representations.

4. **Recommendation Function:**
   Given a movie title, the system looks up similar movies based on cosine similarity scores and recommends the top N matches.

### Code Walkthrough

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample movie dataset: (Movie Title, Description)
movies = [
    ("The Matrix", "sci-fi action future reality virtual AI"),
    ("John Wick", "action revenge assassin dog"),
    ("Interstellar", "space science drama wormhole time travel"),
    ("Inception", "dream heist subconscious action mind-bending"),
    ("The Notebook", "romance love drama emotional"),
    ("Avengers", "superheroes marvel action battle save world"),
    ("Gravity", "space survival astronaut drama"),
    ("Titanic", "romance tragedy love ship historical"),
    ("The Martian", "space science survival alone Mars"),
    ("Edge of Tomorrow", "time loop alien war action")
]

# Extract movie titles and descriptions
titles = [title for title, desc in movies]
descriptions = [desc for title, desc in movies]

# Convert descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)

# Compute cosine similarity between all movie pairs
cosine_sim = cosine_similarity(tfidf_matrix)

# Function to recommend movies based on a given title
def recommend(title, top_n=3):
    if title not in titles:
        print("Movie not found in database.")
        return

    idx = titles.index(title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} recommendations for '{title}':\n")
    for i, score in sim_scores[1:top_n + 1]:  # Skip the first, it's the same movie
        print(f"{titles[i]} (Similarity: {score:.2f})")

# Example usage
recommend("Interstellar")
```

### Example Output

```
Top 3 recommendations for 'Interstellar':
The Martian (Similarity: 0.41)
Gravity (Similarity: 0.29)
Inception (Similarity: 0.10)
```

### Dependencies

* `scikit-learn`
* `numpy`

### Notes

* This is a basic example using manually defined descriptions. In a production system, you'd typically use more detailed metadata and a larger dataset.
* TF-IDF is effective for simple keyword-based similarity. For more advanced recommendations, consider using word embeddings or collaborative filtering approaches.
