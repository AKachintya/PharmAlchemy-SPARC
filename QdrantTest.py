# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:08:34 2025

@author: Akach
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import qdrant_client
from qdrant_client.models import PointStruct, Distance, VectorParams, CollectionConfig

def generate_embeddings(texts, model_name="allenai/biomed_roberta_base"):
    """
    Generate embeddings using BioMed-RoBERTa for a given list of text inputs.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize and convert to tensors
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Run through model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the last hidden state (embedding)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    
    return embeddings.numpy()

def process_csv(file_path):
    """
    Read CSV file and process text data for embedding generation.
    """
    df = pd.read_csv(file_path)
    
    # Identify a relevant text column
    possible_text_cols = ["gene", "interaction_type", "score","drug_name"]
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    
    if text_col is None:
        raise ValueError(f"CSV file must contain a text-related column. Found columns: {df.columns}")
    
    texts = df[text_col].astype(str).tolist()
    embeddings = generate_embeddings(texts)
    
    return texts, embeddings

def store_embeddings_qdrant(collection_name="disease_embeddings", file_path="C:/Users/Akach/Downloads/DGIdb_2_3_25.csv"):
    """
    Store embeddings in Qdrant.
    """
    client = qdrant_client.QdrantClient("localhost", port=6333)
    
    # Create collection if it doesn't exist
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    texts, embeddings = process_csv(file_path)
    points = [PointStruct(id=i, vector=emb.tolist(), payload={"text": texts[i]}) for i, emb in enumerate(embeddings)]
    
    client.upsert(collection_name=collection_name, points=points)
    print("Embeddings stored successfully.")

def search_qdrant(query_text, collection_name="disease_embeddings", model_name="allenai/biomed_roberta_base", top_k=5):
    """
    Search for the most similar diseases in Qdrant.
    """
    client = qdrant_client.QdrantClient("localhost", port=6333)
    
    query_embedding = generate_embeddings([query_text])[0]
    
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    return [(hit.payload["text"], hit.score) for hit in search_results]

# Example test
if __name__ == "__main__":
    store_embeddings_qdrant()
    query = "stomach cancer"
    results = search_qdrant(query)
    print("Search results:", results)
