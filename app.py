from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained pipeline
# Ensure rain_model.pkl is in the same directory
try:
    model = joblib.load('rain_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert data into a DataFrame (the model expects specific column names)
        # Features: Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure
        input_df = pd.DataFrame([data])
        
        # Make prediction
        # 0 = No Rain, 1 = Rain (based on typical Logistic Regression outputs)
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1] # Probability of "Rain"
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': round(float(probability[0]), 4),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'fail'}), 400

if __name__ == '__main__':
    app.run()