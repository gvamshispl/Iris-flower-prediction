from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Loading the Scikit-learn 1.6.1 SVC model
model = joblib.load("model (1).pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']
        # The model expects exactly 4 features
        input_data = np.array([data], dtype=np.float64)
        
        prediction = model.predict(input_data)
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Port 7860 is the Hugging Face Space default
    app.run(host='0.0.0.0', port=7860)