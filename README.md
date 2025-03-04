# 🍷 Bacchus-with-chakna
This project is a machine learning-based recommendation system that suggests the best drink for a given snack (or vice versa). The model is trained on a dataset of food and wine pairings and is deployed on AWS EC2 with a Flask-based API and frontend.

🚀 Features

✅ Predicts best drink for a snack and vice versa

✅ Provides pairing notes for recommendations

✅ Users can give feedback (1-10 rating) to improve model
✅ Frontend built using Flask & HTML/CSS/JS
✅ Stores feedback data in AWS S3

🏗️ Project Architecture
Backend API (app.py)

-Runs on Flask and serves recommendations
-Fetches and stores model files in AWS S3
-Uses RandomForestClassifier for predictions
-Collects feedback & retrains model

Frontend UI (app_frontend.py)

-Built with Flask + HTML + CSS + JS
-Sends requests to backend API for recommendations
-Allows users to rate pairings

AWS Services Used

-EC2: Hosting backend & frontend
-S3: Storing model files and feedback data
-API Gateway (Optional): To make API public
