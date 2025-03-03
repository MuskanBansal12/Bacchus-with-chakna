from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load models, encoders, and dataset
try:
    drink_model = joblib.load("drink_recommendation_model.pkl")
    snack_model = joblib.load("snack_recommendation_model.pkl")
    pairing_notes = joblib.load("pairing_notes.pkl")
    drink_encoder = joblib.load("drink_encoder.pkl")
    snack_encoder = joblib.load("snack_encoder.pkl")
    df = pd.read_csv("processed_dataset.csv")

    # Convert column types to standard Python integers
    df["Snack Name"] = df["Snack Name"].astype(int)
    df["Beverage Name"] = df["Beverage Name"].astype(int)

    print("✅ Models and Data Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    exit(1)

# Feedback storage file
FEEDBACK_FILE = "feedback.csv"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input provided!"}), 400

        result = {}

        if "snack" in data:
            snack_name = data["snack"]

            # Encode the snack
            try:
                input_snack = int(snack_encoder.transform([snack_name])[0])
            except ValueError:
                return jsonify({
                    "error": f"Snack '{snack_name}' not found!",
                    "available_snacks": list(snack_encoder.classes_)
                }), 400

            print(f"✅ Encoded Value for {snack_name}: {input_snack}")

            # Find matching rows
            input_features = df[df["Snack Name"] == input_snack].drop(columns=["Beverage Name"])

            if input_features.empty:
                return jsonify({
                    "error": f"No matching features found for snack '{snack_name}'",
                    "encoded_snack": int(input_snack),
                    "unique_snacks": [int(s) for s in df["Snack Name"].unique()]
                }), 400

            # Predict drink
            prediction = int(drink_model.predict([input_features.iloc[0]])[0])
            recommended_drink = drink_encoder.inverse_transform([prediction])[0]

            pairing_note = pairing_notes.get((recommended_drink, snack_name), "No pairing note available")

            result = {
                "recommended_drink": recommended_drink,
                "pairing_note": pairing_note,
                "message": "Please provide feedback by sending a POST request to /feedback with {'snack': snack_name, 'drink': recommended_drink, 'rating': 1-10}"
            }

        elif "drink" in data:
            drink_name = data["drink"]

            # Encode the drink
            try:
                input_drink = int(drink_encoder.transform([drink_name])[0])
            except ValueError:
                return jsonify({
                    "error": f"Drink '{drink_name}' not found!",
                    "available_drinks": list(drink_encoder.classes_)
                }), 400

            print(f"✅ Encoded Value for {drink_name}: {input_drink}")

            # Find matching rows
            input_features = df[df["Beverage Name"] == input_drink].drop(columns=["Snack Name"])

            if input_features.empty:
                return jsonify({
                    "error": f"No matching features found for drink '{drink_name}'",
                    "encoded_drink": int(input_drink),
                    "unique_drinks": [int(s) for s in df["Beverage Name"].unique()]
                }), 400

            # Predict snack
            prediction = int(snack_model.predict([input_features.iloc[0]])[0])
            recommended_snack = snack_encoder.inverse_transform([prediction])[0]

            pairing_note = pairing_notes.get((drink_name, recommended_snack), "No pairing note available")

            result = {
                "recommended_snack": recommended_snack,
                "pairing_note": pairing_note,
                "message": "Please provide feedback by sending a POST request to /feedback with {'snack': recommended_snack, 'drink': drink_name, 'rating': 1-10}"
            }

        else:
            return jsonify({"error": "Provide either 'snack' or 'drink' as input"}), 400

        return jsonify(result)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def collect_feedback():
    try:
        data = request.json
        if "snack" not in data or "drink" not in data or "rating" not in data:
            return jsonify({"error": "Provide 'snack', 'drink', and 'rating' (1-10)"}), 400

        snack = data["snack"]
        drink = data["drink"]
        rating = int(data["rating"])

        if rating < 1 or rating > 10:
            return jsonify({"error": "Rating must be between 1 and 10"}), 400

        # Save feedback
        feedback_entry = pd.DataFrame([[snack, drink, rating]], columns=["Snack", "Drink", "Rating"])
        if not os.path.exists(FEEDBACK_FILE):
            feedback_entry.to_csv(FEEDBACK_FILE, index=False)
        else:
            feedback_entry.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)

        # **Check if enough feedback is collected to retrain**
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        if len(feedback_df) >= 10:
            retrain_model()  # **Automatically retrain when 10 feedback entries are collected**
            return jsonify({"message": "Feedback recorded successfully! Model retrained with latest feedback."})

        return jsonify({"message": "Feedback recorded successfully!"})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

def retrain_model():
    """ Retrains the model when enough feedback data is collected """
    try:
        feedback_df = pd.read_csv(FEEDBACK_FILE)

        # Encode feedback
        feedback_df["Snack"] = snack_encoder.transform(feedback_df["Snack"])
        feedback_df["Drink"] = drink_encoder.transform(feedback_df["Drink"])

        # Retrain models using feedback
        X = feedback_df.drop(columns=["Rating"])
        y_drink = feedback_df["Drink"]
        y_snack = feedback_df["Snack"]

        X_train, X_test, y_train, y_test = train_test_split(X, y_drink, test_size=0.2, random_state=42)
        drink_model.fit(X_train, y_train)
        joblib.dump(drink_model, "drink_recommendation_model.pkl")

        X_train, X_test, y_train, y_test = train_test_split(X, y_snack, test_size=0.2, random_state=42)
        snack_model.fit(X_train, y_train)
        joblib.dump(snack_model, "snack_recommendation_model.pkl")

        print("✅ Model retrained successfully with feedback data!")

    except Exception as e:
        print(f"❌ Error retraining model: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
