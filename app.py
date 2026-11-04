from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Create Flask application
# --------------------------------------------------

app = Flask(__name__)

# --------------------------------------------------
# File paths
# --------------------------------------------------

MODEL_FILE = "revenue_model.pkl"
DATA_FILE = "retail_store_data.csv"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print("Error loading model:", e)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

try:
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
except Exception as e:
    df = pd.DataFrame()
    print("Error loading dataset:", e)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Revenue Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from JSON request
        data = request.get_json()

        # Read input values
        store_size = float(data["store_size"])
        customers = float(data["customers"])
        marketing = float(data["marketing"])
        inventory = float(data["inventory"])
        employees = float(data["employees"])
        online_orders = float(data["online_orders"])
        discount = float(data["discount"])

        # Check if model exists
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model file revenue_model.pkl was not found."
            }), 500

        # Create input dataframe
        input_data = pd.DataFrame([{
            "Store_Size": store_size,
            "Customers": customers,
            "Marketing_Spend": marketing,
            "Inventory_Level": inventory,
            "Employees": employees,
            "Online_Orders": online_orders,
            "Discount_Percentage": discount
        }])

        # Make prediction
        prediction = model.predict(input_data)

        # Get predicted revenue
        predicted_revenue = float(prediction[0])

        # Return result
        return jsonify({
            "success": True,
            "prediction": round(predicted_revenue, 2)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# --------------------------------------------------
# Dataset API
# --------------------------------------------------

@app.route("/data")
def data():

    try:

        if df.empty:
            return jsonify({
                "error": "Dataset could not be loaded."
            }), 500

        # Return first 50 rows
        return jsonify(
            df.head(50).to_dict(orient="records")
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

@app.route("/dataset-info")
def dataset_info():

    try:

        if df.empty:
            return jsonify({
                "error": "Dataset could not be loaded."
            }), 500

        return jsonify({
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "dataset_loaded": not df.empty
    })


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
        use_debugger=False
    )