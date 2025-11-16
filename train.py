import pandas as pd
import numpy as np
import pickle
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# ---------------------------------------------
# Helper functions
# ---------------------------------------------

def count_items(x):
    """Parse JSON-like columns and return number of elements."""
    try:
        items = ast.literal_eval(x)
        if isinstance(items, list):
            return len(items)
        return 0
    except:
        return 0


# ---------------------------------------------
# Load data
# ---------------------------------------------

df = pd.read_csv("data/tmdb_5000_movies.csv")


# ---------------------------------------------
# Cleaning
# ---------------------------------------------

# homepage -> binary flag, then drop
df["has_homepage"] = df["homepage"].notnull().astype(int)
df = df.drop(columns=["homepage"])

# drop text columns we don't use
df = df.drop(columns=["tagline", "overview"])

# impute runtime
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

# release_date
df = df.dropna(subset=["release_date"])
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year


# ---------------------------------------------
# Feature engineering
# ---------------------------------------------

df["genres_count"] = df["genres"].apply(count_items)
df["keywords_count"] = df["keywords"].apply(count_items)
df["production_companies_count"] = df["production_companies"].apply(count_items)
df["production_countries_count"] = df["production_countries"].apply(count_items)
df["spoken_languages_count"] = df["spoken_languages"].apply(count_items)

df["budget_log"] = np.log1p(df["budget"])
df["revenue_log"] = np.log1p(df["revenue"])

df["vote_ratio"] = df["vote_average"] / (df["vote_count"] + 1)

# target variable
df["success"] = (df["revenue"] > 2 * df["budget"]).astype(int)


# ---------------------------------------------
# Final feature selection
# ---------------------------------------------

feature_cols = [
    "budget_log", "runtime", "popularity", "vote_average", "vote_count",
    "vote_ratio", "genres_count", "keywords_count",
    "production_companies_count", "production_countries_count",
    "spoken_languages_count", "has_homepage", "release_year"
]

df_clean = df[feature_cols + ["success"]].dropna()

X = df_clean.drop(columns=["success"])
y = df_clean["success"]


# ---------------------------------------------
# Train/Val/Test Split
# ---------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# ---------------------------------------------
# Scaling
# ---------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------
# Final Model: Random Forest (best params)
# ---------------------------------------------

best_params = {
    "n_estimators": 600,
    "min_samples_split": 10,
    "max_depth": 10,
}

rf_final = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_train, y_train)

# Validation AUC (optional)
y_pred_val = rf_final.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_pred_val)
print("Validation AUC:", auc_val)

# Test AUC (optional)
y_pred_test = rf_final.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_pred_test)
print("Test AUC:", auc_test)


# ---------------------------------------------
# Save model + scaler
# ---------------------------------------------

with open("model.bin", "wb") as f_out:
    pickle.dump((rf_final, scaler), f_out)

print("model.bin saved successfully.")