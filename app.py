import os
import logging
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langserve import add_routes
import csv
import chardet
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading environment variables...")
# Load environment variables
load_dotenv()

# Load the CSV file with error handling for encoding issues
def load_csv_file(file_path):
    logger.info(f"Detecting encoding for file: {file_path}")
    # Detect encoding
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    logger.info(f"Detected encoding: {encoding}")

    # Possible delimiters
    delimiters = [';']

    for delimiter in delimiters:
        try:
            logger.info(f"Trying to load CSV with delimiter: '{delimiter}' and encoding: '{encoding}'")
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                header=None,  # Adjust if there's a header
                on_bad_lines='skip',
                engine='python'  # Use the python engine for better error handling
            )
            logger.info(f"Successfully loaded CSV with delimiter: '{delimiter}'")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV with delimiter '{delimiter}': {e}")
    raise Exception(f"Failed to load the file: {file_path} with detected encoding and available delimiters.")

logger.info("Loading CSV file...")
df = load_csv_file("ecoles_francaises.csv")

logger.info("Preparing data for the vector store...")
# Prepare the data for the vector store
texts = df.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1).tolist()
metadatas = df.to_dict('records')
logger.info("Data preparation complete.")

logger.info("Creating OpenAI embeddings...")
# Create the vector store
embeddings = OpenAIEmbeddings()
logger.info("Embeddings created.")

logger.info("Building the FAISS vector store...")
vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
logger.info("Vector store created.")

logger.info("Creating retriever...")
# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
logger.info("Retriever created.")

logger.info("Setting up the custom prompt template...")
# Create a custom prompt template in French
template = """Vous êtes un assistant IA spécialisé dans la réponse aux questions sur les écoles françaises basées sur des données CSV.
Utilisez les éléments de contexte suivants pour répondre à la question à la fin.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

{context}

Question: {question}
Réponse: """

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)
logger.info("Prompt template set up.")

logger.info("Initializing the RetrievalQA chain...")
# Create the chain
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)
logger.info("RetrievalQA chain initialized.")

class RAGInput(BaseModel):
    question: str = Field(description="La question à poser sur les données CSV des écoles françaises")
    chat_history: Optional[List[str]] = Field(default_factory=list, description="Historique de chat pour le contexte")

class RAGOutput(BaseModel):
    answer: str = Field(description="La réponse à la question basée sur les données CSV des écoles françaises")

logger.info("Setting up the FastAPI app...")
app = FastAPI(
    title="RAG Spécialisé pour les Données CSV des Écoles Françaises",
    version="1.0",
    description="Un système RAG alimenté par LangChain pour interroger les données CSV sur les écoles françaises",
)
logger.info("FastAPI app setup complete.")

@app.post("/rag", response_model=RAGOutput)
async def rag_endpoint(input: RAGInput):
    logger.info(f"Received question: {input.question}")
    result = chain({"query": input.question})
    logger.info("Generated answer.")
    return RAGOutput(answer=result["result"])

# Add routes for the RAG chain
logger.info("Adding routes for the RAG chain...")
add_routes(
    app,
    chain.with_types(input_type=RAGInput, output_type=RAGOutput),
    path="/rag-chain",
)
logger.info("Routes added.")

if __name__ == "__main__":
    logger.info("Starting the Uvicorn server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
