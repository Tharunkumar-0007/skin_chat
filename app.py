from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import os
import numpy as np
import re
import torch
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration for SQLAlchemy
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/userdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Image Classifier Model
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model_path = 'D:/saved/union/image_classifier_model.h5' #C:/Users/tharu/Desktop/Demo/union/image_classifier_model.h5
model = load_model(model_path)
# Define class labels for image 

with open('diseases.json', 'r') as f:
    class_labels = json.load(f)

# Global variables for the QA chain and other components
qa_chain = None
question_count = 0  # Counter for the number of questions asked

# Prompt template for QA bot
custom_prompt_template = """Answer the following question using the given context.
Context: {context}
Question: {question}
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    """Load the language model."""
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
    return llm

from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.cached(timeout=300, key_prefix='faq_cache')
def get_faq_response(question):
    return qa_chain({'query': question})

def retrieval_qa_chain(llm, prompt, db):
    """Create a RetrievalQA chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

def initialize_qa_bot():
    """Initialize the QA bot and store it in a global variable."""
    global qa_chain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})

    try:
        print("Loading FAISS database...")
        faiss_path = os.getenv('FAISS_DB_PATH', 'vectorstores/db_faiss')
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS database loaded successfully.")
    except FileNotFoundError:
        print("FAISS index not found. Please create the FAISS index first.")
        return None
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None

    try:
        
        print("Loading LLM...")
        llm = load_llm()
        print("LLM loaded successfully.")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return None

    try:
        print("Setting custom prompt...")
        qa_prompt = set_custom_prompt()
        print("Creating QA chain...")
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
        print("QA chain created successfully.")
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in the 'templates' folder

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


def is_medical_query(query):
    """Use a classifier to determine if the question is medical-related."""
    labels = ['medical', 'non-medical']
    result = classifier(query, labels)
    return result['labels'][0] == 'medical'

@app.route('/ask', methods=['POST'])
def ask_question():
    global question_count
    user_input = request.form['query']
    username = request.form.get('username')  # Assuming you pass the username in the form data
   
    if not is_valid_query(user_input):
        if username:
            app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": "Nothing matched. Please enter a valid query."})
   
    # Check if the query is medical-related
    if not is_medical_query(user_input):
        app.logger.info(f'Non-medical question by {username}: {user_input}')
        return jsonify({"response": "Not medical-related"})
   
    if qa_chain is None:
        if username:
            app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": "Failed to initialize QA bot."})
   
    try:
        res = qa_chain({'query': user_input})
        answer = res.get("result", "No answer found.")
        question_count += 1
        app.logger.info(f'Question count: {question_count}')
        if username:
            app.logger.info(f'Question asked by {username}: {user_input}')
        return jsonify({"response": answer})
    except Exception as e:
        if username:
            app.logger.error(f'Error processing the query by {username}: "{user_input}" - Error: {e}')
        return jsonify({"response": f"Error processing the query: {e}"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_probability = np.max(predictions)

        threshold = 0.6

        if predicted_probability < threshold:
            predicted_label = 'Healthy Skin or Not a Valid Disease Image'
        else:
            predicted_label = class_labels.get(str(predicted_class), 'Unknown')

        return jsonify({'predicted_class': predicted_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def is_valid_query(query):
    """Check if the query is valid."""
    if not query or query.isspace():
        return False
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
    return True

if __name__ == '__main__':
    initialize_qa_bot()  # Initialize the QA bot when the app starts
    app.run(host='0.0.0.0', debug=True, port=5000)  

