# ingest_data.py

import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

import pandas as pd

# For demonstration, we'll assume you're using OpenAI embeddings.
# If using Google Vertex AI embeddings, you'd replace this with the correct class/wrapper.
EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"  # Example from OpenAI

UN_VECTORSTORE_DIR = "un_food_index"         # Folder to store the Chroma DB for UN data
CLINICAL_VECTORSTORE_DIR = "clinical_index"  # Folder to store the Chroma DB for Clinical data

def ingest_un_food_data(pdf_path: str, db_dir: str):
    print(f"Ingesting UN Food PDF data from: {pdf_path}")
    # Read the PDF
    with pdfplumber.open(pdf_path) as pdf:
        text_pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_pages.append((page_num, text))

    # Convert to Documents
    docs = []
    for page_num, page_text in text_pages:
        docs.append(
            Document(
                page_content=page_text,
                metadata={"source": "un_food_security", "page": page_num}
            )
        )
    
    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"UN Food Security vectorstore created at: {db_dir}")


def ingest_clinical_data(csv_path: str, db_dir: str):
    print(f"Ingesting Clinical CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert each row to a Document
    # We'll transform row data into a single text field for embedding
    docs = []
    for idx, row in df.iterrows():
        row_text = (
            f"NCT Number: {row.get('NCT Number','')}\n"
            f"Study Title: {row.get('Study Title','')}\n"
            f"Study URL: {row.get('Study URL','')}\n"
            f"Study Status: {row.get('Study Status','')}\n"
            f"Conditions: {row.get('Conditions','')}\n"
            f"Interventions: {row.get('Interventions','')}\n"
            f"Sponsor: {row.get('Sponsor','')}\n"
            f"Collaborators: {row.get('Collaborators','')}\n"
            f"Phases: {row.get('Phases','')}\n"
            f"Enrollment: {row.get('Enrollment','')}\n"
            f"Study Type: {row.get('Study Type','')}\n"
        )
        docs.append(
            Document(
                page_content=row_text,
                metadata={"source": "clinical_studies", "row_index": idx}
            )
        )

    # Split if needed (some rows might be large, but typically short enough)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME, openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"Clinical vectorstore created at: {db_dir}")


if __name__ == "__main__":
    # Provide paths
    un_pdf_path = os.path.join("data", "un_food_security.pdf")
    clinical_csv_path = os.path.join("data", "clinical_studies.csv")

    # Ingest
    ingest_un_food_data(un_pdf_path, UN_VECTORSTORE_DIR)
    ingest_clinical_data(clinical_csv_path, CLINICAL_VECTORSTORE_DIR)
