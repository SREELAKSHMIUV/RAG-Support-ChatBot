from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

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

print("Support RAG Chatbot Ready (type 'exit' to quit)")

while True:

    query = input("\nAsk a support question: ")

    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(query, k=1)

    context = docs[0].page_content

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful customer support assistant. Answer only using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )

    answer = response.choices[0].message.content

    print("\nSupport Agent:")
    print(answer)