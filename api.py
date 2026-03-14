from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from llm.llm_factory import get_llm

import os


# Load environment variables
load_dotenv()

# Initialize LLM from factory
llm = get_llm()


app = FastAPI()


# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class Query(BaseModel):
    question: str


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS vector database
vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)


# System prompt
system_prompt = """
You are a friendly AI support chatbot.

Your job is to help users solve issues based ONLY on the provided context.

Rules:
- Only use the given context.
- Do NOT create answers outside the context.
- Give short solution steps.
- Use bullet points.
"""


@app.post("/chat")
def chat(query: Query):

    question = query.question.lower()

    # ---------- Identity Questions ----------
    if "who are you" in question or "what are you" in question:
        return {"answer": "I'm an AI support chatbot designed to help users solve common issues."}

    if "what do you do" in question:
        return {"answer": "I help users by answering support-related questions using the available knowledge base."}

    # ---------- Retrieve Relevant Documents ----------
    docs_with_scores = vectorstore.similarity_search_with_score(query.question, k=3)

    relevant_docs = []

    for doc, score in docs_with_scores:
        if score < 0.5:
            relevant_docs.append(doc)

    # If no relevant context found
    if len(relevant_docs) == 0:
        return {"answer": "Sorry, I don't have information about that."}

    # Combine context
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Create prompt for LLM
    user_prompt = f"""
Context:
{context}

User Question:
{query.question}

Provide a short helpful answer.
"""

    # Generate answer using LLM abstraction
    answer = llm.generate(system_prompt, user_prompt)

    return {"answer": answer}