# app.py
from flask import Flask, render_template, request, jsonify
import re
from flask_cors import CORS
from skin import predict_skin_disease 
from chat import ask_question  

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['query']
    if not user_input.strip() or not re.search(r'[a-zA-Z0-9]', user_input):
        return jsonify({"response": "Nothing matched. Please enter a valid query."})

    response = ask_question(user_input)
    return jsonify({"response": response})

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Call the prediction function from skin.py
        result = predict_skin_disease(file)

        # If the result contains a message (like invalid image), return that
        if 'message' in result:
            return jsonify({'error': result['message']}), 400

        app.logger.info(f"Prediction result: {result}")
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000, use_reloader=False)
