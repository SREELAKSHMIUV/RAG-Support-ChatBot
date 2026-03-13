from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load document
loader = TextLoader("data/support_knowledge.txt")
documents = loader.load()

print("Documents loaded:", len(documents))

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))

# Step 3: Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")

# Step 4: Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

print("Vector database created")

# Step 5: Save vector database locally
vectorstore.save_local("vectorstore")

print("Vector database saved successfully")