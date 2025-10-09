from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ====================== ENV ======================
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.parquet")
OCCURRENCES_PATH = os.environ.get("OCCURRENCES_PATH", "occurrences.parquet")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # "*" = aberto


# =================== APP & CORS ==================
app = Flask(__name__)

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*"
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=False)


# ================ MODEL (lazy load) ==============
_model = None
_model_error = None


def ensure_model():
    """Carrega o modelo sob demanda."""
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            _model_error = str(e)


# ============== HISTORY (eager load) =============
_history = None
_history_error = None

try:
    if HISTORY_PATH.endswith(".parquet"):
        _history = pd.read_parquet(HISTORY_PATH)
    else:
        _history = pd.read_csv(HISTORY_PATH)

    _history["year"] = _history["year"].astype(int)
    _history["month"] = _history["month"].astype(int)
    _history["crime_count"] = _history["crime_count"].astype(float)

except Exception as e:
    _history_error = str(e)


# ========= OCCURRENCES (eager) p/ heatmap =========
_occ = None
_occ_error = None

try:
    if OCCURRENCES_PATH.endswith(".parquet"):
        _occ = pd.read_parquet(OCCURRENCES_PATH)
    else:
        _occ = pd.read_csv(OCCURRENCES_PATH)

    if "data_ocorrencia" in _occ.columns:
        _occ["data_ocorrencia"] = pd.to_datetime(_occ["data_ocorrencia"], errors="coerce")
        _occ["year"] = _occ["data_ocorrencia"].dt.year

    req_cols = {"descricao_modus_operandi", "tipo_crime"}
    missing = req_cols - set(_occ.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

except Exception as e:
    _occ_error = str(e)


# ===================== HELPERS ====================
def prev_year_month(year: int, month: int) -> tuple[int, int]:
    """Calcula o ano/mês anterior."""
    dt = datetime.date(int(year), int(month), 1)
    prev = dt.replace(day=1) - datetime.timedelta(days=1)
    return prev.year, prev.month


def get_observed(bairro: str, year: int, month: int):
    """Retorna crime_count observado para (bairro, ano, mês) ou None."""
    if _history is None:
        return None
    m = (_history["bairro"] == bairro) & (_history["year"] == year) & (_history["month"] == month)
    return float(_history.loc[m, "crime_count"].iloc[0]) if m.any() else None


def infer_previous_count(bairro: str, year: int, month: int):
    """Descobre previous_month_crime_count; retorna (valor, fonte)."""
    if _history is None or len(_history) == 0:
        return (None, None)

    py, pm = prev_year_month(year, month)

    val = get_observed(bairro, py, pm)
    if val is not None:
        return (val, "history")

    hb = _history[_history["bairro"] == bairro]
    if not hb.empty:
        return (float(hb["crime_count"].median()), "bairro_median")

    return (float(_history["crime_count"].median()), "global_median")


# ===================== ENDPOINTS ==================
@app.get("/health")
def health():
    ensure_model()
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_error": _model_error,
        "history_loaded": _history is not None,
        "history_error": _history_error,
        "occurrences_loaded": _occ is not None,
        "occurrences_error": _occ_error
    })


@app.post("/predict")
def predict():
    """Predição pontual; infere previous_month_crime_count se não enviado."""
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503

    payload = request.get_json(silent=True) or {}

    miss = [k for k in ["bairro", "year", "month"] if k not in payload]
    if miss:
        return jsonify({"error": f"Campos faltando: {miss}. Envie ao menos ['bairro','year','month']."}), 400

    bairro = str(payload["bairro"])
    year = int(payload["year"])
    month = int(payload["month"])

    if "previous_month_crime_count" in payload and payload["previous_month_crime_count"] is not None:
        prev_used = float(payload["previous_month_crime_count"])
    else:
        prev_used, _ = infer_previous_count(bairro, year, month)
        if prev_used is None:
            return jsonify({
                "error": "previous_month_crime_count ausente e histórico indisponível.",
                "tip": "Defina HISTORY_PATH e faça deploy."
            }), 503

    X = pd.DataFrame([{
        "bairro": bairro,
        "year": year,
        "month": month,
        "previous_month_crime_count": float(prev_used)
    }])

    yhat = float(_model.predict(X)[0])

    return jsonify({
        "input": {"bairro": bairro, "year": year, "month": month},
        "prediction": {"crime_count_pred": yhat}
    })


@app.get("/year/summary")
def year_summary():
    """Retorna meses observed/non_observed (previsão) em JSON simplificado."""
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503
    if _history is None:
        return jsonify({"error": "Histórico indisponível.", "history_error": _history_error}), 503

    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"error": "Parâmetro 'year' é obrigatório (ex.: ?year=2025)"}), 400

    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]
    if not bairros:
        bairros = sorted(_history["bairro"].dropna().unique().tolist())

    vt_param = request.args.get("value_type", default=None)
    allowed_types = {v.strip() for v in vt_param.split(",") if v.strip()} if vt_param else {"observed", "non_observed"}

    rows = []
    last_value_cache = {b: {} for b in bairros}

    for bairro in bairros:
        for m in range(1, 13):
            obs = get_observed(bairro, year, m)

            if obs is not None:
                rows.append({"bairro": bairro, "month": m, "value": float(obs), "value_type": "observed"})
                last_value_cache[bairro][m] = float(obs)
                continue

            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
            else:
                prev_used, _ = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    continue

            Xp = pd.DataFrame([{
                "bairro": bairro,
                "year": year,
                "month": m,
                "previous_month_crime_count": float(prev_used)
            }])

            yhat = float(_model.predict(Xp)[0])

            rows.append({"bairro": bairro, "month": m, "value": yhat, "value_type": "non_observed"})
            last_value_cache[bairro][m] = yhat

    rows = [r for r in rows if r["value_type"] in allowed_types]

    return jsonify({
        "year": year,
        "bairros": bairros,
        "data": rows
    })


@app.get("/pca")
def pca_endpoint():
    """Retorna pca1/pca2 + rótulos de cluster (KMeans)."""
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503
    if _history is None:
        return jsonify({"error": "Histórico indisponível.", "history_error": _history_error}), 503

    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"error": "Parâmetro 'year' é obrigatório (ex.: ?year=2025)"}), 400

    k = request.args.get("k", default=3, type=int)
    if k < 2 or k > 10:
        return jsonify({"error": "Parâmetro 'k' deve estar entre 2 e 10.", "k": k}), 400

    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]
    if not bairros:
        bairros = sorted(_history["bairro"].dropna().unique().tolist())

    rows = []
    last_value_cache = {b: {} for b in bairros}

    for bairro in bairros:
        for m in range(1, 13):
            obs = get_observed(bairro, year, m)

            if obs is not None:
                rows.append({"bairro": bairro, "month": m, "value": float(obs)})
                last_value_cache[bairro][m] = float(obs)
                continue

            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
            else:
                prev_used, _ = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    continue

            Xp = pd.DataFrame([{
                "bairro": bairro,
                "year": year,
                "month": m,
                "previous_month_crime_count": float(prev_used)
            }])

            yhat = float(_model.predict(Xp)[0])

            rows.append({"bairro": bairro, "month": m, "value": yhat})
            last_value_cache[bairro][m] = yhat

    if len(rows) < k:
        return jsonify({"error": "Amostras insuficientes para clusterização/PCA.", "n_samples": len(rows), "k": k}), 400

    dfp = pd.DataFrame(rows).sort_values(["bairro", "month"]).reset_index(drop=True)

    theta = 2 * np.pi * (dfp["month"].astype(float) - 1) / 12.0
    dfp["month_sin"] = np.sin(theta)
    dfp["month_cos"] = np.cos(theta)

    X = dfp[["value", "month_sin", "month_cos"]].to_numpy(dtype=float)

    X_scaled = StandardScaler().fit_transform(X)

    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)

    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    return jsonify({
        "pca1": X_pca[:, 0].astype(float).tolist(),
        "pca2": X_pca[:, 1].astype(float).tolist(),
        "cluster": labels.astype(int).tolist()
    })


@app.get("/heatmap")
def heatmap():
    """Crosstab bruto de descricao_modus_operandi x tipo_crime (contagem)."""
    if _occ is None:
        return jsonify({
            "error": "Dataset de ocorrências indisponível.",
            "occurrences_error": _occ_error,
            "tip": "Defina OCCURRENCES_PATH para CSV/Parquet."
        }), 503

    year = request.args.get("year", type=int)

    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]

    index_top = request.args.get("index_top", type=int)
    columns_top = request.args.get("columns_top", type=int)

    df = _occ.copy()

    if year is not None and "year" in df.columns:
        df = df[df["year"] == year]

    if bairros:
        if "bairro" not in df.columns:
            return jsonify({"error": "Coluna 'bairro' não existe no dataset para aplicar filtro."}), 400
        df = df[df["bairro"].isin(bairros)]

    need = ["descricao_modus_operandi", "tipo_crime"]
    if any(c not in df.columns for c in need):
        return jsonify({"error": f"Dataset não possui as colunas necessárias: {need}"}), 400

    df = df.dropna(subset=need)

    if index_top:
        top_rows = df["descricao_modus_operandi"].value_counts().head(index_top).index
        df = df[df["descricao_modus_operandi"].isin(top_rows)]

    if columns_top:
        top_cols = df["tipo_crime"].value_counts().head(columns_top).index
        df = df[df["tipo_crime"].isin(top_cols)]

    if df.empty:
        return jsonify({
            "params": {"year": year, "bairros": bairros, "index_top": index_top, "columns_top": columns_top},
            "index": [], "columns": [], "data": [], "value_kind": "count"
        })

    ct = pd.crosstab(df["descricao_modus_operandi"], df["tipo_crime"])

    row_order = ct.sum(axis=1).sort_values(ascending=False).index
    col_order = ct.sum(axis=0).sort_values(ascending=False).index
    ct = ct.loc[row_order, col_order].astype(int)

    out = ct.to_dict(orient="split")

    return jsonify({
        "params": {"year": year, "bairros": bairros, "index_top": index_top, "columns_top": columns_top},
        "index": out["index"],
        "columns": out["columns"],
        "data": out["data"],
        "value_kind": "count"
    })


# =================== MAIN (dev) ===================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
