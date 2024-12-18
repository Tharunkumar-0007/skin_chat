# chat.py
import os
import torch
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import pipeline

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

def ask_question(user_input):
    """Function to handle a user query."""
    if qa_bot.is_medical_query(user_input):
        if qa_bot.qa_chain:
            try:
                response = qa_bot.qa_chain.invoke({'query': user_input}).get("result", "No answer found.")
                qa_bot.question_count += 1
                return response
            except Exception as e:
                return f"Error processing the query: {e}"
        else:
            return "Failed to initialize QA bot."
    else:
        return "Not medical-related"
