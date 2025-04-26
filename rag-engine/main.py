from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_mongodb import MongoDBAtlasVectorSearch

from langchain_ollama import OllamaEmbeddings

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (adjust origins as needed in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
collection = client[os.getenv("MONGO_DB")][os.getenv("MONGO_COLLECTION")]

# Route to handle questions
@app.post("/query")
async def ask_question(request: Request):
    data = await request.json()
    question = data["question"]

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        index_name="medical_index"
    )


    qa = RetrievalQA.from_chain_type(
        llm=Ollama(model="gemma:2b"),
        retriever=vectorstore.as_retriever()
    )

    result = qa.run(question)
    return {"answer": result}

# Run the app (if run as script)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
