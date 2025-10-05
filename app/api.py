from flask import Flask, request, jsonify
import os, joblib, pandas as pd
import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "history.parquet")

app = Flask(__name__)

# Carrega modelo (simples). Se preferir, troque por lazy-load.
model = joblib.load(MODEL_PATH)

# Carrega histórico agregado (observado)
history = None
history_error = None
try:
    if HISTORY_PATH.endswith(".parquet"):
        history = pd.read_parquet(HISTORY_PATH)
    else:
        history = pd.read_csv(HISTORY_PATH)

    history["year"] = history["year"].astype(int)
    history["month"] = history["month"].astype(int)
    history["crime_count"] = history["crime_count"].astype(float)
except Exception as e:
    history_error = str(e)

def prev_year_month(year: int, month: int) -> tuple[int, int]:
    dt = datetime.date(int(year), int(month), 1)
    prev = dt.replace(day=1) - datetime.timedelta(days=1)
    return prev.year, prev.month

def as_bool(v, default=True) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def get_observed(bairro: str, year: int, month: int):
    if history is None:
        return None
    m = (history["bairro"] == bairro) & (history["year"] == year) & (history["month"] == month)
    if not m.any():
        return None
    return float(history.loc[m, "crime_count"].iloc[0])

def infer_previous_count(bairro: str, year: int, month: int):
    """
    Tenta descobrir o valor do mês anterior.
    Retorna (valor, fonte) com fonte ∈ {"history","bairro_median","global_median", None}
    """
    if history is None:
        return (None, None)

    py, pm = prev_year_month(year, month)
    # 1) histórico do mês anterior
    val = get_observed(bairro, py, pm)
    if val is not None:
        return (val, "history")

    # 2) fallback: mediana do bairro
    hb = history[history["bairro"] == bairro]
    if not hb.empty:
        return (float(hb["crime_count"].median()), "bairro_median")

    # 3) fallback global
    return (float(history["crime_count"].median()), "global_median")

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "history_loaded": history is not None,
        "history_error": history_error
    })

@app.post("/predict")
def predict():
    """
    Continua aceitando previous_month_crime_count explicitamente,
    mas se não vier, a API tenta inferir do histórico.
    """
    payload = request.get_json(silent=True) or {}
    required_base = ["bairro", "year", "month"]
    miss = [k for k in required_base if k not in payload]
    if miss:
        return jsonify({"error": f"Campos faltando: {miss}. Envie ao menos {required_base}."}), 400

    bairro = str(payload["bairro"])
    year = int(payload["year"])
    month = int(payload["month"])

    prev_used = None
    prev_source = "provided"
    if "previous_month_crime_count" in payload and payload["previous_month_crime_count"] is not None:
        prev_used = float(payload["previous_month_crime_count"])
    else:
        prev_used, prev_source = infer_previous_count(bairro, year, month)
        if prev_used is None:
            return jsonify({
                "error": "previous_month_crime_count ausente e histórico indisponível.",
                "tip": "Defina HISTORY_PATH para history.parquet e faça deploy com esse arquivo."
            }), 503

    X = pd.DataFrame([{
        "bairro": bairro,
        "year": year,
        "month": month,
        "previous_month_crime_count": prev_used,
    }])

    yhat = float(model.predict(X)[0])
    return jsonify({
        "input": {
            "bairro": bairro,
            "year": year,
            "month": month,
            "previous_month_crime_count": prev_used
        },
        "prediction": {
            "crime_count_pred": yhat
        },
        "context": {"previous_count_source": prev_source}
    })

@app.get("/year/summary")
def year_summary():
    """
    Ex: GET /year/summary?year=2025&bairro=Boa%20Viagem&include_predictions=true
    - year (obrigatório)
    - bairro (opcional, pode repetir ou ser CSV)
    - include_predictions (opcional, default true)
    """
    if history is None:
        return jsonify({"error": "Histórico indisponível no servidor.", "history_error": history_error}), 503

    # parâmetros
    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"error": "Parâmetro 'year' é obrigatório (ex.: ?year=2025)"}), 400

    include_predictions = as_bool(request.args.get("include_predictions"), default=True)

    # bairros: aceitar lista (repetido) ou CSV
    bairros = request.args.getlist("bairro")
    if len(bairros) == 1 and "," in bairros[0]:
        bairros = [b.strip() for b in bairros[0].split(",") if b.strip()]
    if not bairros:
        bairros = sorted(history["bairro"].dropna().unique().tolist())

    # grade 12 meses para cada bairro
    rows = []
    # guardamos o "valor do mês anterior" usado durante a construção,
    # para permitir cadeia de previsões dentro do próprio ano.
    last_value_cache = {b: {} for b in bairros}  # {bairro: {m: value}}

    for bairro in bairros:
        for m in range(1, 13):
            observed = get_observed(bairro, year, m)
            if observed is not None:
                rows.append({
                    "bairro": bairro,
                    "month": m,
                    "observed": observed,
                    "predicted": None,
                    "value": observed,
                    "value_type": "observed",
                    "prev_used": None,
                    "prev_source": None
                })
                last_value_cache[bairro][m] = observed
                continue

            # sem observado
            if not include_predictions:
                rows.append({
                    "bairro": bairro,
                    "month": m,
                    "observed": None,
                    "predicted": None,
                    "value": None,
                    "value_type": "missing",
                    "prev_used": None,
                    "prev_source": None
                })
                continue

            # previsão: precisamos do previous_month_crime_count
            # 1) se mês anterior no mesmo ano já tem value (observed ou predicted), use-o
            if (m - 1) in last_value_cache[bairro]:
                prev_used = last_value_cache[bairro][m - 1]
                prev_source = "same_year_chain"
            else:
                # 2) senão, busque no histórico (mês real anterior, inclusive ano anterior)
                prev_used, prev_source = infer_previous_count(bairro, year, m)
                if prev_used is None:
                    # como fallback extremo, pule previsão
                    rows.append({
                        "bairro": bairro,
                        "month": m,
                        "observed": None,
                        "predicted": None,
                        "value": None,
                        "value_type": "missing",
                        "prev_used": None,
                        "prev_source": None
                    })
                    continue

            X = pd.DataFrame([{
                "bairro": bairro,
                "year": year,
                "month": m,
                "previous_month_crime_count": float(prev_used),
            }])
            yhat = float(model.predict(X)[0])

            rows.append({
                "bairro": bairro,
                "month": m,
                "observed": None,
                "predicted": yhat,
                "value": yhat,
                "value_type": "predicted",
                "prev_used": float(prev_used),
                "prev_source": prev_source
            })
            last_value_cache[bairro][m] = yhat

    # summary
    df = pd.DataFrame(rows)

    totals = (
        df.groupby("bairro")
          .agg(
              observed_sum=("observed", "sum"),
              predicted_sum=("predicted", "sum"),
          )
          .fillna(0.0)
          .reset_index()
    )
    totals["year_total"] = totals["observed_sum"] + totals["predicted_sum"]

    top_bairros = (
        totals.sort_values("year_total", ascending=False)
              .head(10)[["bairro", "year_total"]]
              .to_dict(orient="records")
    )

    # YoY simples (somente observado do ano anterior para transparência)
    yoy_list = []
    prev_year = year - 1
    for bairro in bairros:
        prev_obs = history[(history["bairro"] == bairro) & (history["year"] == prev_year)]
        prev_year_total = float(prev_obs["crime_count"].sum()) if not prev_obs.empty else 0.0
        curr = totals[totals["bairro"] == bairro].iloc[0]
        curr_total = float(curr["year_total"])
        yoy_list.append({
            "bairro": bairro,
            "prev_year": prev_year,
            "prev_year_total": prev_year_total,
            "curr_year_total": curr_total,
            "delta_abs": curr_total - prev_year_total,
            "delta_pct": ( (curr_total - prev_year_total) / prev_year_total * 100.0 ) if prev_year_total > 0 else None
        })

    payload = {
        "year": year,
        "filters": {"bairros": bairros},
        "meta": {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "include_predictions": include_predictions,
            "notes": [
                "observed = dado histórico; predicted = estimativa do modelo.",
                "prev_source indica a origem do previous_month_crime_count.",
            ]
        },
        "data": rows,
        "summary": {
            "totals_by_bairro": totals.to_dict(orient="records"),
            "top_bairros_by_year_total": top_bairros,
            "yoy": yoy_list
        }
    }
    return jsonify(payload)
