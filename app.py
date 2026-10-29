import os
import json
from flask import Flask, render_template, request, jsonify
from predict import DigitPredictor

app = Flask(__name__, static_folder='static', template_folder='templates')
predictor = DigitPredictor(model_path='models/cnn_digit_model.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        result = predictor.predict(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    history_file = 'models/training_history.json'
    history_data = {}
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history_data = json.load(f)
            
    info = {
        'model_name': 'DigitCNN (Dual Conv Blocks + Dropout + BatchNorm)',
        'framework': 'PyTorch',
        'is_loaded': predictor.loaded,
        'parameters': predictor.model.count_parameters(),
        'input_shape': '1 x 28 x 28',
        'num_classes': 10,
        'history': history_data
    }
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Handwritten Digit Recognition CNN Server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
