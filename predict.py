import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------------------------------------------
# Load model and scaler
# ----------------------------------------------------------

with open("model.bin", "rb") as f_in:
    model, scaler = pickle.load(f_in)

# ----------------------------------------------------------
# FastAPI setup
# ----------------------------------------------------------

app = FastAPI(title="Movie Success Prediction API")


# ----------------------------------------------------------
# Request schema
# ----------------------------------------------------------

class MovieFeatures(BaseModel):
    budget_log: float
    runtime: float
    popularity: float
    vote_average: float
    vote_count: float
    vote_ratio: float
    genres_count: float
    keywords_count: float
    production_companies_count: float
    production_countries_count: float
    spoken_languages_count: float
    has_homepage: int
    release_year: int


# ----------------------------------------------------------
# Prediction endpoint
# ----------------------------------------------------------

@app.post("/predict")
def predict_success(features: MovieFeatures):

    # Convert input into model-ready array
    movie_array = np.array([
        [
            features.budget_log,
            features.runtime,
            features.popularity,
            features.vote_average,
            features.vote_count,
            features.vote_ratio,
            features.genres_count,
            features.keywords_count,
            features.production_companies_count,
            features.production_countries_count,
            features.spoken_languages_count,
            features.has_homepage,
            features.release_year
        ]
    ])

    # Scale with StandardScaler (if needed)
    movie_scaled = scaler.transform(movie_array)

    # Predict probability
    proba = model.predict_proba(movie_scaled)[0, 1]

    return {
        "success_probability": float(proba)
    }