from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
import joblib
import pandas as pd

# ====== Config por ENV ======
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.parquet")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # deixe "*" para acesso aberto

# ====== App & CORS ======
app = Flask(__name__)
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*"
CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=False)

# ====== Carregamento lazy do modelo ======
_model = None
_model_error = None
def ensure_model():
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            _model_error = str(e)

# ====== Histórico (observado) carregado ao subir ======
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

# ====== Helpers ======
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

# ====== Endpoints ======
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
    Retorna meses 'observed' e 'non_observed' (preenchidos por previsão) num JSON simplificado.

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

    # bairros
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
                    "value": observed, "value_type": "observed"
                })
                last_value_cache[bairro][m] = observed
                continue

            # previsões forcadas para não observados
            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
            else:
                prev_used, _ = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    # não conseguimos prever este mês; simplesmente não incluímos
                    continue

            X = pd.DataFrame([{
                "bairro": bairro,
                "year": year,
                "month": m,
                "previous_month_crime_count": float(prev_used),
            }])
            yhat = float(_model.predict(X)[0])

            rows.append({
                "bairro": bairro, "month": m,
                "value": yhat, "value_type": "non_observed"
            })
            last_value_cache[bairro][m] = yhat

    # aplica filtro de value_type
    rows = [r for r in rows if r["value_type"] in allowed_types]

    # resposta simplificada
    return jsonify({
        "year": year,
        "bairros": bairros,
        "data": rows
    })
