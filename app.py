from flask import Flask, request, jsonify, render_template_string
from catboost import CatBoostRegressor
import pandas as pd

app = Flask(__name__)

# ================= Load CatBoost model safely =================
try:
    model = CatBoostRegressor()
    model.load_model("catboost_salary_model.cbm")  # Make sure this file is in the same folder
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# ================= Home route with HTML form =================
@app.route('/')
def home():
    html = '''
    <h2>Salary Prediction Form</h2>
    <form action="/predict" method="post">
        Department: <input type="text" name="Department" value="IT"><br><br>
        Job Title: <input type="text" name="Job_Title" value="Data Scientist"><br><br>
        Education Level: <input type="text" name="Education_Level" value="Master"><br><br>
        Experience (Years): <input type="number" name="Experience_Years" value="5"><br><br>
        <input type="submit" value="Predict Salary">
    </form>
    '''
    return render_template_string(html)

# ================= Prediction route =================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model not loaded"

    try:
        # Get form data
        data = request.form.to_dict()

        # Ensure numeric columns are correct type
        if "Experience_Years" in data:
            data["Experience_Years"] = float(data["Experience_Years"])

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict salary
        prediction = model.predict(input_df)[0]

        return f"<h3>Predicted Salary: {round(float(prediction), 2)}</h3>"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ================= Run the Flask app =================
if __name__ == "__main__":
    print("📌 Running Flask server...")
    app.run(debug=True)
