from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing objects
model = joblib.load("optimized_loan_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert input values to correct types
        input_data = {
            "person_age": int(data["person_age"]),
            "person_income": float(data["person_income"]),
            "person_home_ownership": data["person_home_ownership"],
            "person_emp_length": float(data["person_emp_length"]),
            "loan_intent": data["loan_intent"],
            "loan_grade": data["loan_grade"],
            "loan_amnt": float(data["loan_amnt"]),
            "loan_int_rate": float(data["loan_int_rate"]),
            "loan_percent_income": float(data["loan_percent_income"]),
            "cb_person_default_on_file": data["cb_person_default_on_file"],
            "cb_person_cred_hist_length": float(data["cb_person_cred_hist_length"]),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical values
        categorical_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
        for col in categorical_cols:
            if col in label_encoders:
                # Handle unseen values
                input_df[col] = input_df[col].apply(lambda x: label_encoders[col].transform([x])[0] 
                                                    if x in label_encoders[col].classes_ else -1)

        # Standardize numerical values
        input_df = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Loan Approved ✅" if prediction == 0 else "Loan Not Approved ❌"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
