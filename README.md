# movie-success-prediction

Midterm Project — DataTalksClub Machine Learning Zoomcamp

Machine learning project to predict whether a movie will be a commercial success based on its metadata (budget, popularity, runtime, genres, etc.). Developed as part of the DataTalksClub Machine Learning Zoomcamp midterm project.

---

## 📌 Problem Description

Movie production companies invest millions of dollars into films, but not all movies become profitable.  
This project aims to **predict whether a movie will be a commercial success** based on its metadata available before or shortly after release.

We define **success = 1** when:
    
    revenue > 2 × budget


This project explores:

- What features influence movie success
- How well different ML models can predict profitability
- How to deploy a predictive model as an API

---

## 📁 Dataset

We used the **TMDB 5000 Movies Dataset**.  
It includes metadata such as:

- Budget  
- Revenue  
- Genres  
- Keywords  
- Popularity  
- Runtime  
- Production companies / countries  
- Vote average / count  
- Release date  

Files used:

    data/tmdb_5000_movies.csv


The dataset is included in the repository so the project is fully reproducible.

---

## 🧹 Data Cleaning & Feature Engineering

Steps applied:

- Removed unused text fields (`overview`, `tagline`)
- Converted `homepage` into a binary flag
- Filled missing `runtime` with median
- Extracted `release_year`
- Counted list-like fields:
  - `genres_count`
  - `keywords_count`
  - `production_companies_count`
  - `production_countries_count`
  - `spoken_languages_count`
- Applied log transformation:
  - `budget_log`
  - `revenue_log`
- Created target variable:

    success = 1 if revenue > 2 * budget else 0


Final feature set:

    [
        "budget_log", "runtime", "popularity", "vote_average", "vote_count",
        "vote_ratio", "genres_count", "keywords_count",
        "production_companies_count", "production_countries_count",
        "spoken_languages_count", "has_homepage", "release_year"
    ]


---

## 📊 Model Training & Evaluation

Train/Val/Test split:

- **60% / 20% / 20%**
- Stratified to preserve success ratio

Models trained:

✔ Logistic Regression  
✔ Random Forest  
✔ XGBoost  

Hyperparameter tuning:

- RandomSearchCV for Random Forest  
- Manual parameter grid for XGBoost  

### 🎯 Best Model  
**Random Forest Classifier** with:
    
    {
        "n_estimators": 600,
        "max_depth": 10,
        "min_samples_split": 10
    }


### ⭐ Final Test Score  
**AUC = 0.8688**

---

## 🧪 Reproducibility

### 📌 Install dependencies:

    pip install -r requirements.txt


### 📌 Train the model:

    python train.py

This will generate:

    model.bin

---

## 🌐 Running the API

### Start the API:

    uvicorn predict:app --reload --port 8000

### Open Swagger UI:

http://127.0.0.1:8000/docs

---

## 📝 Example Request JSON

```json
{
  "budget_log": 16.0,
  "runtime": 120,
  "popularity": 35.0,
  "vote_average": 6.8,
  "vote_count": 1200,
  "vote_ratio": 0.005,
  "genres_count": 3,
  "keywords_count": 5,
  "production_companies_count": 2,
  "production_countries_count": 1,
  "spoken_languages_count": 1,
  "has_homepage": 1,
  "release_year": 2015
}


## 🐳 Docker Usage

### 🔧 Build image

docker build -t movie-success-api .

### Run container:

docker run -p 8000:8000 movie-success-api

### Access API:

http://127.0.0.1:8000/docs








  




