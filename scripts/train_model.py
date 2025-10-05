import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

CSV_PATH = os.environ.get("DATASET_PATH", "data/raw/dataset_ocorrencias_delegacia_5.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.parquet")

def build_aggregates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # garante datetime
    df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"], errors="coerce")
    df["year"] = df["data_ocorrencia"].dt.year
    df["month"] = df["data_ocorrencia"].dt.month

    # agrega contagem por bairro/ano/mês
    agg = (
        df.groupby(["bairro", "year", "month"], as_index=False)
          .size()
          .rename(columns={"size": "crime_count"})
          .sort_values(["bairro", "year", "month"])
          .reset_index(drop=True)
    )
    return agg

def train_and_save(agg: pd.DataFrame, model_path: str):
    # cria lag de 1 mês por bairro para treino
    train = agg.copy()
    train["previous_month_crime_count"] = train.groupby("bairro")["crime_count"].shift(1)
    train = train.dropna(subset=["previous_month_crime_count"]).reset_index(drop=True)
    train["previous_month_crime_count"] = train["previous_month_crime_count"].astype(float)

    X = train[["bairro", "year", "month", "previous_month_crime_count"]]
    y = train["crime_count"].astype(float)

    pre = ColumnTransformer(
        [("bairro_ohe", OneHotEncoder(handle_unknown="ignore"), ["bairro"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X, y)

    joblib.dump(pipe, model_path)
    print(f"Modelo salvo em: {model_path}")

def save_history(agg: pd.DataFrame, history_path: str):
    # salvamos SÓ o histórico "observado" (sem a coluna de lag)
    if history_path.endswith(".parquet"):
        agg.to_parquet(history_path, index=False)
    else:
        agg.to_csv(history_path, index=False)
    print(f"Histórico salvo em: {history_path}")

if __name__ == "__main__":
    aggregates = build_aggregates(CSV_PATH)
    train_and_save(aggregates, MODEL_PATH)
    save_history(aggregates[["bairro", "year", "month", "crime_count"]], HISTORY_PATH)
