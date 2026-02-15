from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ===============================
# Load trained model artifacts
# ===============================
artifact = joblib.load("student_dropout_rf_best.pkl")

model = artifact["model"]
label_encoders = artifact["label_encoders"]

# ===============================
# Initialize Flask app
# ===============================
app = Flask(__name__)

# ===============================
# Home route (health check)
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "Student Dropout Prediction API is running"

# ===============================
# Prediction route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical columns
        for col, le in label_encoders.items():
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400
            df[col] = le.transform(df[col].astype(str))

        # Ensure correct column order
        df = df[model.feature_names_in_]

        # Predict probability
        probability = model.predict_proba(df)[0][1]
        label = "Yes" if probability >= 0.5 else "No"

        return jsonify({
            "dropout": label,
            "probability": round(float(probability * 100), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
