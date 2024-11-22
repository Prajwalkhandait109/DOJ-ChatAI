from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
import fitz  # PyMuPDF for PDF text extraction

# Initialize the model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Folder containing your law PDFs
data_folder = 'data'  # Replace with your actual folder path

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Read all PDF files in the data folder
law_documents = []
for filename in os.listdir(data_folder):
    if filename.endswith('.pdf'):  # Assuming PDF files for documents
        file_path = os.path.join(data_folder, filename)
        pdf_text = extract_text_from_pdf(file_path)
        law_documents.append(pdf_text)

# Compute embeddings for the documents
document_embeddings = embedding_model.encode(law_documents)

# Save the embeddings and documents
with open('vectorstore_embeddings.npy', 'wb') as f:
    np.save(f, document_embeddings)

with open('vectorstore_content.pkl', 'wb') as f:
    pickle.dump(law_documents, f)

print("Embeddings and documents saved successfully!")
