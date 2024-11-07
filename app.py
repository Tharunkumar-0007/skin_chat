from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import uuid
import os
import numpy as np
import re
import torch
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)

class ImageClassifier:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)
        with open(labels_path, 'r') as f:
            self.class_labels = json.load(f)
    
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array

    def predict(self, image_path, threshold=0.6):
        if os.path.exists(image_path):
            try:
                img_array = self.preprocess_image(image_path)
                predictions = self.model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_probability = np.max(predictions)

                if predicted_probability < threshold:
                    return 'Healthy Skin or Not a Valid Disease Image'
                else:
                    return self.class_labels.get(str(predicted_class), 'Unknown')
            except Exception as e:
                return f"Prediction error: {e}"
        else:
            return 'Invalid image path'

# Load image classifier
classifier_model_path = 'D:/project/union/image_classifier_model.h5'
diseases_labels_path = 'diseases.json'
image_classifier = ImageClassifier(classifier_model_path, diseases_labels_path)

class QABot:
    def __init__(self):
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.question_count = 0
        self.initialize_qa_bot()

    def is_medical_query(self, query):
        labels = ['medical', 'non-medical']
        result = self.classifier(query, labels)
        return result['labels'][0] == 'medical'

    def initialize_qa_bot(self):
        faiss_path = os.getenv('FAISS_DB_PATH', 'vectorstores/db_faiss')
        if os.path.exists(faiss_path):
            try:
                print("Loading FAISS database...")
                db = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS database loaded successfully.")
            except FileNotFoundError:
                print("FAISS index not found. Please create the FAISS index first.")
                return
            except Exception as e:
                print(f"Error loading FAISS database: {e}")
                return

            try:
                print("Loading LLM...")
                llm = CTransformers(
                    model="TheBloke/llama-2-7b-chat-GGML",
                    model_type="llama",
                    max_new_tokens=128,
                    temperature=0.7,
                    n_gpu_layers=8,
                    n_threads=24,
                    n_batch=1000,
                    load_in_8bit=True,
                    num_beams=1,
                    max_length=256,
                    clean_up_tokenization_spaces=False
                )
                print("LLM loaded successfully.")
                prompt_template = PromptTemplate(
                    template="""Answer the following question using the given context.
                                Context: {context}
                                Question: {question}
                                Helpful answer:
                             """,
                    input_variables=["context", "question"]
                )
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 1}),
                    chain_type_kwargs={"prompt": prompt_template},
                    return_source_documents=False
                )
                print("QA chain created successfully.")
            except Exception as e:
                print(f"Error initializing QA bot: {e}")

qa_bot = QABot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.form['query']

    if not user_input.strip() or not re.search(r'[a-zA-Z0-9]', user_input):
        return jsonify({"response": "Nothing matched. Please enter a valid query."})

    if qa_bot.is_medical_query(user_input):
        if qa_bot.qa_chain:
            try:
                response = qa_bot.qa_chain({'query': user_input}).get("result", "No answer found.")
                qa_bot.question_count += 1
                return jsonify({"response": response})
            except Exception as e:
                return jsonify({"response": f"Error processing the query: {e}"})
        else:
            return jsonify({"response": "Failed to initialize QA bot."})
    else:
        return jsonify({"response": "Not medical-related"})

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif'}


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        try:
            # Create a unique filename
            filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(os.path.dirname(__file__), filename)
            file.save(file_path)

            # Assuming `image_classifier.predict(file_path)` is your model prediction function
            predicted_label = image_classifier.predict(file_path)

            # Clean up saved file after prediction
            os.remove(file_path)
            return jsonify({'predicted_class': predicted_label}), 200
        except Exception as e:
            return jsonify({'error': f"Failed to process the file: {str(e)}"}), 500
    else:
        return jsonify({'error': 'Invalid file format. Only .png, .jpg, or .jpeg allowed.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000,use_reloader=False)
