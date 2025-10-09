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

# ========= Config por ENV =========
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.parquet")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # "*" = aberto para qualquer origem

# ========= App & CORS =========
app = Flask(__name__)
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*"
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=False)

# ========= Modelo (lazy load) =========
_model = None
_model_error = None
def ensure_model():
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            _model_error = str(e)

# ========= Histórico observado (carregado ao subir) =========
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

# ========= Helpers =========
def prev_year_month(year: int, month: int) -> tuple[int, int]:
    dt = datetime.date(int(year), int(month), 1)
    prev = dt.replace(day=1) - datetime.timedelta(days=1)
    return prev.year, prev.month

def get_observed(bairro: str, year: int, month: int):
    if _history is None:
        return None
    m = (_history["bairro"] == bairro) & (_history["year"] == year) & (_history["month"] == month)
    if not m.any():
        return None
    return float(_history.loc[m, "crime_count"].iloc[0])

def infer_previous_count(bairro: str, year: int, month: int):
    """
    Descobre previous_month_crime_count.
    Retorna (valor, fonte) onde fonte ∈ {"history","bairro_median","global_median", None}
    """
    if _history is None or len(_history) == 0:
        return (None, None)

    py, pm = prev_year_month(year, month)

    # 1) Histórico do mês anterior
    val = get_observed(bairro, py, pm)
    if val is not None:
        return (val, "history")

    # 2) Fallback: mediana do bairro
    hb = _history[_history["bairro"] == bairro]
    if not hb.empty:
        return (float(hb["crime_count"].median()), "bairro_median")

    # 3) Fallback global
    return (float(_history["crime_count"].median()), "global_median")

# ========= Endpoints =========
@app.get("/health")
def health():
    ensure_model()
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_error": _model_error,
        "history_loaded": _history is not None,
        "history_error": _history_error,
    })

@app.post("/predict")
def predict():
    """
    Predição pontual. Aceita previous_month_crime_count explicitamente;
    se não vier, a API infere do histórico.
    """
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503

    payload = request.get_json(silent=True) or {}
    required_base = ["bairro", "year", "month"]
    miss = [k for k in required_base if k not in payload]
    if miss:
        return jsonify({"error": f"Campos faltando: {miss}. Envie ao menos {required_base}."}), 400

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
                "tip": "Defina HISTORY_PATH para history.parquet e faça deploy com esse arquivo."
            }), 503

    X = pd.DataFrame([{
        "bairro": bairro,
        "year": year,
        "month": month,
        "previous_month_crime_count": float(prev_used),
    }])
    yhat = float(_model.predict(X)[0])

    return jsonify({
        "input": {"bairro": bairro, "year": year, "month": month},
        "prediction": {"crime_count_pred": yhat}
    })

@app.get("/year/summary")
def year_summary():
    """
    Retorna meses 'observed' (histórico) e 'non_observed' (previsão) num JSON simplificado.

    Query:
      - year (obrigatório) -> int
      - bairro (opcional; repetir ?bairro=... ou CSV)
      - value_type (opcional; 'observed', 'non_observed' ou CSV com ambos)

    Resposta:
    {
      "year": 2025,
      "bairros": ["Boa Viagem", ...],
      "data": [
        {"bairro":"Boa Viagem","month":1,"value":98,"value_type":"observed"},
        {"bairro":"Boa Viagem","month":10,"value":15.18,"value_type":"non_observed"}
      ]
    }
    """
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503
    if _history is None:
        return jsonify({"error": "Histórico indisponível no servidor.", "history_error": _history_error}), 503

    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"error": "Parâmetro 'year' é obrigatório (ex.: ?year=2025)"}), 400

    # bairros (lista ou CSV)
    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]
    if not bairros:
        bairros = sorted(_history["bairro"].dropna().unique().tolist())

    # filtro value_type
    vt_param = request.args.get("value_type", default=None)
    if vt_param:
        allowed_types = {v.strip() for v in vt_param.split(",") if v.strip()}
    else:
        allowed_types = {"observed", "non_observed"}  # padrão: ambos

    rows = []
    last_value_cache: dict[str, dict[int, float]] = {b: {} for b in bairros}

    for bairro in bairros:
        for m in range(1, 13):
            observed = get_observed(bairro, year, m)
            if observed is not None:
                rows.append({
                    "bairro": bairro, "month": m,
                    "value": float(observed), "value_type": "observed"
                })
                last_value_cache[bairro][m] = float(observed)
                continue

            # Sem observado: prever (forçado)
            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
            else:
                prev_used, _ = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    # sem como estimar: pula (não entra em data)
                    continue

            Xp = pd.DataFrame([{
                "bairro": bairro, "year": year, "month": m,
                "previous_month_crime_count": float(prev_used),
            }])
            yhat = float(_model.predict(Xp)[0])

            rows.append({
                "bairro": bairro, "month": m,
                "value": yhat, "value_type": "non_observed"
            })
            last_value_cache[bairro][m] = yhat

    # aplica filtro de value_type
    rows = [r for r in rows if r["value_type"] in allowed_types]

    return jsonify({
        "year": year,
        "bairros": bairros,
        "data": rows
    })

@app.get("/pca")
def pca_endpoint():
    """
    Ex:
      GET /pca?year=2025&bairro=Boa%20Viagem         # k=3 (padrão)
      GET /pca?year=2025&bairro=Boa%20Viagem&k=3

    Retorna:
      { "pca1": [...], "pca2": [...], "cluster": [0,2,1,...] }

    Observação:
      - Ordem estável dos pontos: ordenado por (bairro, month).
      - Clusters alinhados com pca1/pca2 (mesmo índice).
    """
    ensure_model()
    if _model is None:
        return jsonify({"error": f"Modelo indisponível: {_model_error}"}), 503
    if _history is None:
        return jsonify({"error": "Histórico indisponível no servidor.", "history_error": _history_error}), 503

    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"error": "Parâmetro 'year' é obrigatório (ex.: ?year=2025)"}), 400

    k = request.args.get("k", default=3, type=int)
    if k < 2 or k > 10:
        return jsonify({"error": "Parâmetro 'k' deve estar entre 2 e 10.", "k": k}), 400

    # bairros (lista ou CSV)
    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]
    if not bairros:
        bairros = sorted(_history["bairro"].dropna().unique().tolist())

    # monta 'value' (observado ou previsão) para cada mês, com ordem estável (bairro, month)
    rows = []
    last_value_cache: dict[str, dict[int, float]] = {b: {} for b in bairros}

    for bairro in bairros:
        for m in range(1, 13):
            obs = get_observed(bairro, year, m)
            if obs is not None:
                rows.append({"bairro": bairro, "month": m, "value": float(obs)})
                last_value_cache[bairro][m] = float(obs)
                continue

            # previsão encadeada quando possível
            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
            else:
                prev_used, _ = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    continue

            Xp = pd.DataFrame([{
                "bairro": bairro, "year": year, "month": m,
                "previous_month_crime_count": float(prev_used),
            }])
            yhat = float(_model.predict(Xp)[0])
            rows.append({"bairro": bairro, "month": m, "value": yhat})
            last_value_cache[bairro][m] = yhat

    if len(rows) < k:
        return jsonify({"error": "Amostras insuficientes para clusterização/PCA.", "n_samples": len(rows), "k": k}), 400

    dfp = pd.DataFrame(rows).sort_values(["bairro", "month"]).reset_index(drop=True)

    # Features numéricas para cluster/PCA: value + sazonalidade (sen/cos de mês)
    theta = 2 * np.pi * (dfp["month"].astype(float) - 1) / 12.0
    dfp["month_sin"] = np.sin(theta)
    dfp["month_cos"] = np.cos(theta)

    X = dfp[["value", "month_sin", "month_cos"]].to_numpy(dtype=float)

    # Escala
    X_scaled = StandardScaler().fit_transform(X)

    # ---- KMeans nos dados escalados (k=3 por padrão) ----
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # ---- PCA 2D para visualização (mesmo X_scaled) ----
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return jsonify({
        "pca1": X_pca[:, 0].astype(float).tolist(),
        "pca2": X_pca[:, 1].astype(float).tolist(),
        "cluster": labels.astype(int).tolist()
    })

# ========= Main (para dev local) =========
if __name__ == "__main__":
    # DEV local: python app/api.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
