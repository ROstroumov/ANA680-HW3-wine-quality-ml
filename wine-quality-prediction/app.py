
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# This would be loaded from your trained model
# For now, we'll create a simple placeholder
class WineQualityModel:
    def predict(self, features):
        # Simple rule-based prediction for demo
        # In reality, you'd load your trained model
        alcohol = features[10]  # alcohol content
        if alcohol > 12.0:
            return 8  # high quality
        elif alcohol > 10.5:
            return 6  # medium quality
        else:
            return 4  # low quality

model = WineQualityModel()

@app.route('/')
def home():
    return "Wine Quality Prediction API - Use POST /predict with wine features"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        
        if len(features) != 11:
            return jsonify({'error': 'Expected 11 features for wine quality prediction'})
        
        prediction = model.predict(features)
        return jsonify({
            'prediction': prediction,
            'quality_score': prediction,
            'message': f'Wine quality prediction: {prediction}/10'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
