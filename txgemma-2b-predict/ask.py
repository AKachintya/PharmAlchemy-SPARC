import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
# This points to your folder of .txt files
DATA_PATH = "data"
# This is the name of the folder where the vector database will be stored
DB_PATH = "vector_db"
# This is the model used to create the embeddings (numerical representations of your documents)
EMBED_MODEL = "gemma:7b" 
# This is the model that will answer your questions
LLM_MODEL = "my-txgemma"

def create_vector_db():
    """
    Reads all .txt files from the DATA_PATH, splits them into chunks,
    and stores them in a Chroma vector database.
    """
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print("Creating embeddings and vector store... (This may take a while)")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    print("Vector database created successfully!")
    return db

def load_vector_db():
    """Loads an existing vector database."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db

def main():
    """Main function to run the RAG application."""
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: The '{DATA_PATH}' folder is empty or does not exist.")
        print("Please create the 'data' folder and add your .txt research files to it before running again.")
        return

    if not os.path.exists(DB_PATH):
        print("No existing vector database found. Creating a new one.")
        create_vector_db()

    db = load_vector_db()
    
    llm = Ollama(model=LLM_MODEL)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    print("\n--- AI Drug Research Assistant is Ready ---")
    print(f"--- Using model: {LLM_MODEL} ---")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nAsk your question: ")
        if query.lower() == 'exit':
            break
        
        print("Thinking...")
        result = qa_chain.invoke({"query": query})
        
        print("\n--- Answer ---")
        print(result["result"])
        print("\n--- Sources ---")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    main()