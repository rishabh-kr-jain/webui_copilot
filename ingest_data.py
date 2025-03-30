import os
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Directories to store vector databases
UN_VECTORSTORE_DIR = "un_food_index"
CLINICAL_VECTORSTORE_DIR = "clinical_index"

def ingest_un_food_pdf(pdf_path: str, db_dir: str):
    """
    Reads and chunks a UN Food Security PDF, embeds using Hugging Face, stores in Chroma.
    """
    print(f"📄 Ingesting UN Food PDF: {pdf_path}")
    all_docs = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source": "un_food_security", "page": page_idx}
                )
                all_docs.append(doc)

    if not all_docs:
        print("⚠️ No text found in UN PDF. Skipping ingestion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"✅ UN Food PDF chunked into {len(chunked_docs)} segments.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"📦 UN Food Security vectorstore stored at: {db_dir}")


def ingest_clinical_pdf_custom_parsing(pdf_path: str, db_dir: str):
    """
    Extracts clinical trial data from text-formatted lines in a PDF, embeds with HF, stores in Chroma.
    """
    print(f"📄 Ingesting Clinical Studies PDF: {pdf_path}")
    all_docs = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            for line in lines:
                if not line.startswith("NCT"):
                    continue  # likely not a data row

                try:
                    match_nct = re.match(r"(NCT\d+)", line)
                    nct_number = match_nct.group(1) if match_nct else "Unknown"

                    url_match = re.search(r"(https://clinicaltrials.gov/study/NCT\d+)", line)
                    url = url_match.group(1) if url_match else ""

                    title_start = line.find(nct_number) + len(nct_number)
                    title_end = line.find(url)
                    study_title = line[title_start:title_end].strip()

                    after_url = line[title_end + len(url):]
                    status_match = re.match(r"([A-Z_]+)", after_url)
                    status = status_match.group(1) if status_match else ""

                    rest = after_url[len(status):].strip()

                    intervention_keywords = [
                        "DRUG:", "DEVICE:", "OTHER:", "GENETIC:", "BEHAVIORAL:",
                        "PROCEDURE:", "COMBINATION_PRODUCT:", "BIOLOGICAL:",
                        "DIAGNOSTIC_TEST:", "DIETARY_SUPPLEMENT:"
                    ]
                    int_start = min([rest.find(k) for k in intervention_keywords if k in rest] + [len(rest)])
                    conditions = rest[:int_start].strip()
                    interventions_and_beyond = rest[int_start:].strip()

                    interventions = []
                    for token in interventions_and_beyond.split():
                        if any(token.startswith(k) for k in intervention_keywords):
                            interventions.append(token)
                        else:
                            break
                    intervention_text = " ".join(interventions)
                    remaining = interventions_and_beyond[len(intervention_text):].strip()

                    doc_text = (
                        f"NCT Number: {nct_number}\n"
                        f"Study Title: {study_title}\n"
                        f"Study URL: {url}\n"
                        f"Study Status: {status}\n"
                        f"Conditions: {conditions}\n"
                        f"Interventions: {intervention_text}\n"
                        f"Other Info: {remaining}"
                    )

                    all_docs.append(Document(page_content=doc_text, metadata={"source": "clinical_study_pdf"}))

                except Exception as e:
                    print(f"⚠️ Skipped line due to error: {e}\n{line}")

    if not all_docs:
        print("⚠️ No valid clinical rows parsed.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Clinical data chunked into {len(chunks)} segments.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"📦 Clinical vectorstore stored at: {db_dir}")


if __name__ == "__main__":
    un_pdf_path = "data/un_food_security.pdf"
    clinical_pdf_path = "data/clinical_studies.pdf"

    ingest_un_food_pdf(un_pdf_path, UN_VECTORSTORE_DIR)
    ingest_clinical_pdf_custom_parsing(clinical_pdf_path, CLINICAL_VECTORSTORE_DIR)

    print("✅ Ingestion complete!")
