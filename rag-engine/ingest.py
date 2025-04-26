import fitz
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
collection = db[os.getenv("MONGO_COLLECTION")]

def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def embed():
    text = load_pdf("demo_Medical_book.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name="medical_index"
    )
    print("✅ Embeddings stored")

if __name__ == "__main__":
    embed()
