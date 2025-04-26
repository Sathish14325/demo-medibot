from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

app = FastAPI()

# Pydantic model for request payload
class QueryPayload(BaseModel):
    question: str

# Load and split PDF
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

# Create or load vector store
def get_vectorstore(docs, db_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(db_path)
        return vectordb

# Setup
docs = load_and_split_pdf("diabetes.pdf")
vectordb = get_vectorstore(docs)

llm = OllamaLLM(model="mistral")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# API endpoint
@app.post("/query")
async def ask_question(payload: QueryPayload):
    result = qa.invoke({"query": payload.question})
    return {"response": result}
