import os
import fitz
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DOCS_PATH = "docs"
VECTOR_DB_PATH = "vector_store"

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf8") as f:
                text = f.read()
                # Consider more sophisticated splitting for .txt
                # For this specific FAQ, you might manually define sections or use regex
                # to split by days, as they are clearly delineated.
                # For demonstration, we'll use a slightly more intelligent RecursiveCharacterTextSplitter
                # which can be configured to split by different separators.
                documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def ingest_docs():
    documents = load_documents_from_folder(DOCS_PATH)
    
    # Improved chunking strategy
    # This splitter attempts to split by paragraphs, then sentences, then characters.
    # It maintains some overlap to preserve context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, # Increased chunk size to capture more context per event/day
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""] # Prioritize splitting by double newlines (paragraphs)
    )
    
    texts = []
    for doc in documents:
        # Add a custom metadata field if applicable, e.g., for the 'onsite-week-faq.txt'
        # you could try to extract headings or specific sections for better context
        # For this example, let's keep it simple and just split the existing documents.
        chunks = text_splitter.split_documents([doc])
        texts.extend(chunks)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"Saved vector store to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    ingest_docs()