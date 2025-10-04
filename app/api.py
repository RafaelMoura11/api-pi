from flask import Flask, request, jsonify
import os, joblib, pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
app = Flask(__name__)

# carrega na subida (simples); em prod, considere lazy loading + retry
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    required = ["bairro", "year", "month", "previous_month_crime_count"]
    miss = [k for k in required if k not in payload]
    if miss:
        return jsonify({"error": f"Campos faltando: {miss}"}), 400

    X = pd.DataFrame([{
        "bairro": payload["bairro"],
        "year": int(payload["year"]),
        "month": int(payload["month"]),
        "previous_month_crime_count": float(payload["previous_month_crime_count"]),
    }])

    yhat = float(model.predict(X)[0])
    return jsonify({"input": payload, "prediction": {"crime_count_pred": yhat}})
    
if __name__ == "__main__":
    # DEV: python app/api.py
    app.run(host="0.0.0.0", port=8000, debug=True)
