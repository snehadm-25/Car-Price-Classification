from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load artifacts
model = joblib.load('car_price_model.joblib')
scaler = joblib.load('scaler.joblib')
encoders = joblib.load('encoders.joblib')
processed_data = joblib.load('processed_data.joblib')
feature_names = processed_data['feature_names']

@app.route('/')
def home():
    # Provide categorical options for the dropdowns
    options = {}
    for col, le in encoders.items():
        if col != 'target':
            options[col] = list(le.classes_)
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Prepare input data
        input_list = []
        for feature in feature_names:
            val = data.get(feature)
            if feature in encoders:
                # Handle categorical feature
                le = encoders[feature]
                try:
                    val = le.transform([val])[0]
                except ValueError:
                    val = 0
            else:
                # Handle numeric feature
                val = float(val) if val else 0.0
            input_list.append(val)
        
        # Scale and Predict
        input_array = np.array(input_list).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        
        # Inverse transform target
        target_name = encoders['target'].inverse_transform([prediction])[0]
        
        return render_template('index.html', 
                                   prediction=target_name, 
                                   input_data=data,
                                   options={col: list(le.classes_) for col, le in encoders.items() if col != 'target'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
