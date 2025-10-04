import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

CSV_PATH = os.environ.get("DATASET_PATH", "data/raw/dataset_ocorrencias_delegacia_5.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

def build_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], errors="coerce")
    df["year"] = df["data_ocorrencia"].dt.year
    df["month"] = df["data_ocorrencia"].dt.month

    agg = (
        df.groupby(["bairro", "year", "month"], as_index=False)
          .size()
          .rename(columns={"size": "crime_count"})
          .sort_values(["bairro","year","month"])
          .reset_index(drop=True)
    )

    agg["previous_month_crime_count"] = agg.groupby("bairro")["crime_count"].shift(1)
    agg = agg.dropna(subset=["previous_month_crime_count"]).reset_index(drop=True)
    agg["previous_month_crime_count"] = agg["previous_month_crime_count"].astype(float)
    return agg

def train_and_save(df: pd.DataFrame, path: str):
    X = df[["bairro", "year", "month", "previous_month_crime_count"]]
    y = df["crime_count"].astype(float)

    pre = ColumnTransformer(
        [("bairro_ohe", OneHotEncoder(handle_unknown="ignore"), ["bairro"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    print(f"Modelo salvo em: {path}")

if __name__ == "__main__":
    data = build_dataset(CSV_PATH)
    train_and_save(data, MODEL_PATH)
