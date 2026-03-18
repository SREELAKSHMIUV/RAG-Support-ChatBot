from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from llm.llm_factory import get_llm
from models import Query
from prompts.prompt_loader import load_prompts

import os


# Load environment variables
load_dotenv()

# Initialize LLM
llm = get_llm()

# Load prompts from YAML
prompts = load_prompts()

# Extract system prompt
system_prompt = prompts["system_prompt"]

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/chat")
def chat(query: Query):

    question = query.question.lower()

    # ---------- Identity Questions ----------
    if "who are you" in question or "what are you" in question:
        return {"answer": prompts["identity_who_are_you"]}

    if "what do you do" in question:
        return {"answer": prompts["identity_what_do_you_do"]}

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

    # Create user prompt
    user_prompt = f"""
Context:
{context}

User Question:
{query.question}

Provide a short helpful answer.
"""

    # Generate response using LLM
    answer = llm.generate(system_prompt, user_prompt)

    return {"answer": answer}