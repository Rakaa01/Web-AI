from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

app = Flask(__name__)

# Load data
data_path = os.path.join('data', 'data_gaji.csv')
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    print(f"File not found at {data_path}")

# Load model
model_path = os.path.join('models', 'model_name.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Model not found at {model_path}")

# Function to predict salary
def predict_salary(years_experience):
    features = np.array([[years_experience]])
    prediction = model.predict(features)
    return prediction[0][0]

# Route to render index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    years_experience = float(request.form['years_experience'])
    prediction = predict_salary(years_experience)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
