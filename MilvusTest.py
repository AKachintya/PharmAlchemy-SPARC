# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:49:54 2025

@author: Akach
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

BATCH_SIZE = 10000  # Adjust based on your data size

def generate_embeddings(texts, model_name="allenai/biomed_roberta_base"):
    """
    Generate embeddings using BioMed-RoBERTa for a given list of text inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.numpy()

def process_csv(file_path):
    """
    Read CSV file and process text data for embedding generation.
    """
    df = pd.read_csv(file_path)
    possible_text_cols = ["gene", "interaction_type", "score", "drug_name"]
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    
    if text_col is None:
        raise ValueError(f"CSV file must contain a text-related column. Found columns: {df.columns}")
    
    texts = df[text_col].astype(str).tolist()
    embeddings = generate_embeddings(texts)
    
    return texts, embeddings

def store_embeddings_milvus(collection_name="disease_embeddings", file_path="C:/Users/Akach/Downloads/DGIdb_2_3_25.csv"):
    """
    Store embeddings in Milvus in batches.
    """
    connections.connect("default", host="localhost", port="19530")
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description="Collection for disease embeddings")
    
    collection = Collection(name=collection_name, schema=schema)
    
    texts, embeddings = process_csv(file_path)
    num_records = len(texts)
    
    for i in range(0, num_records, BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        
        data = [batch_texts, batch_embeddings.tolist()]
        collection.insert(data)
        print(f"Inserted batch {i // BATCH_SIZE + 1}/{(num_records // BATCH_SIZE) + 1}")
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

def search_milvus(query_text, collection_name="disease_embeddings", model_name="allenai/biomed_roberta_base", top_k=5):
    """
    Search for the most similar diseases in Milvus.
    """
    connections.connect("default", host="localhost", port="19530")
    
    collection = Collection(name=collection_name)
    
    query_embedding = generate_embeddings([query_text])[0]
    
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    
    return [(hit.entity.get("text"), hit.distance) for hit in results[0]]

if __name__ == "__main__":
    store_embeddings_milvus()
    query = "brain cancer"
    results = search_milvus(query)
    print("Search results:", results)
