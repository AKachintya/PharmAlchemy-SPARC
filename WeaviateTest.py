# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:00:59 2025
@author: Akach
"""

import torch
import pandas as pd
import json
import numpy as np
import weaviate
from weaviate.connect import ConnectionParams  # Correct import for ConnectionParams
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 10000
MODEL_NAME = "dmis-lab/biobert-v1.1"

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def generate_embeddings(texts):
    """Generate embeddings using the selected biomedical model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def process_csv(file_path):
    """Process CSV file with gene-drug interaction data."""
    df = pd.read_csv(file_path)
    required_cols = {"hyperedge_id", "drug_name", "interaction_type", "r2g_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file missing required columns: {required_cols - set(df.columns)}")

    texts, metadata_list = [], []
    for _, row in df.iterrows():
        text = f"{row['hyperedge_id']} - {row['drug_name']} ({row['interaction_type']}), Score: {row['r2g_score']}"
        texts.append(text)
        metadata_list.append({
            "gene": row["hyperedge_id"],
            "drug": row["drug_name"],
            "interaction_type": str(row["interaction_type"]).lower(),
            "score": float(row["r2g_score"]),
            "text": text
        })
    embeddings = generate_embeddings(texts)
    return texts, metadata_list, embeddings

def process_json(file_path):
    """Process JSON file with G2G, R2G, and G2D edges."""
    with open(file_path, "r") as f:
        data = json.load(f)

    texts, metadata_list = [], []
    for edge_id, edge in data["edges"].items():
        source, target, metadata = edge["source"], edge["target"], edge["metadata"]
        edge_type = metadata.get("edge_type", "unknown")
        scores = metadata.get("scores", {})
        interaction_type = metadata.get("interaction_type", "unknown") if edge_type == "R2G" else "N/A"
        score_val = list(scores.values())[0] if scores else 0.0

        text_representation = f"{edge_type} - {source} -> {target} ({interaction_type}), Score: {score_val}"
        texts.append(text_representation)
        metadata_list.append({
            "gene": source,
            "drug": target,
            "interaction_type": interaction_type.lower(),
            "score": float(score_val),
            "text": text_representation,
            "edge_type": edge_type
        })

    embeddings = generate_embeddings(texts)
    return texts, metadata_list, embeddings

# Weaviate Schema Setup
def create_weaviate_class(class_name, client):
    """Creates a Weaviate class if it does not exist."""
    schema = {
        "class": class_name,
        "description": "Unified HetNet embeddings with gene-drug/disease interactions",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "gene", "dataType": ["string"]},
            {"name": "drug", "dataType": ["string"]},
            {"name": "interaction_type", "dataType": ["string"]},
            {"name": "score", "dataType": ["number"]},
            {"name": "edge_type", "dataType": ["string"]}
        ]
    }

    # Check if the class already exists
    if not client.collections.exists(class_name):
        client.collections.create_from_dict(schema)
        print(f"Created Weaviate class '{class_name}'")
    else:
        print(f"Weaviate class '{class_name}' already exists.")

# Store Embeddings in Weaviate
def store_embeddings_weaviate(class_name, file_path, file_type, client):
    """Processes file (JSON/CSV), computes embeddings, and inserts them into Weaviate."""
    try:
        if file_type == "json":
            texts, metadata_list, embeddings = process_json(file_path)
        else:
            texts, metadata_list, embeddings = process_csv(file_path)

        num_records = len(texts)
        print(f"Processing {num_records} records...")

        with client.batch as batch:
            for i in range(num_records):
                props = {
                    "text": metadata_list[i]["text"],
                    "gene": metadata_list[i]["gene"],
                    "drug": metadata_list[i]["drug"],
                    "interaction_type": metadata_list[i]["interaction_type"],
                    "score": metadata_list[i]["score"],
                    "edge_type": metadata_list[i].get("edge_type", "unknown")
                }
                vector = embeddings[i].tolist()
                batch.add_data_object(
                    data_object=props,
                    class_name=class_name,
                    vector=vector
                )

                if i % BATCH_SIZE == 0 and i != 0:
                    print(f"Inserted {i} records...")

        print("Data insertion complete.")
    except Exception as e:
        print(f"Error storing embeddings: {e}")

# Search in Weaviate
def search_weaviate(query_text, class_name, client, top_k=5):
    """Query Weaviate using nearVector search."""
    try:
        query_vector = generate_embeddings([query_text])[0].tolist()
        
        response = (
            client.query
            .get(class_name, ["text", "gene", "drug", "interaction_type", "score", "edge_type"])
            .with_near_vector({"vector": query_vector})
            .with_limit(top_k)
            .do()
        )
        
        results = response["data"]["Get"][class_name]
        return results
    except Exception as e:
        print(f"Error searching Weaviate: {e}")
        return []

# Main Execution
if __name__ == "__main__":
    # Connect to Weaviate
    connection_params = ConnectionParams(
        http={
            "host": "localhost",  # HTTP host
            "port": 8080,         # HTTP port
            "secure": False      # Set to True if using HTTPS
        },
        grpc={
            "host": "localhost",  # gRPC host (optional, but required by the schema)
            "port": 50051,        # gRPC port (optional, but required by the schema)
            "secure": False      # Set to True if using secure gRPC
        }
    )
    client = weaviate.WeaviateClient(connection_params)
    client.connect()  # Explicitly connect to the Weaviate server

    class_name = "UnifiedHetNet"
    create_weaviate_class(class_name, client)

    # Store embeddings from a JSON file
    store_embeddings_weaviate(class_name, "C:/Users/Akach/Downloads/AlzheimersUnifiedHetNet.json", "json", client)

    # Example queries
    queries = [
        "drugs that increase the biological activity of the APP gene",
        "agonist drugs for APP gene",
        "inhibitor drugs for ADRB1 gene",
        "modulator drugs for ADRB2 gene"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = search_weaviate(query, class_name, client, top_k=5)
        for res in results:
            print(f"Gene: {res['gene']} | Drug: {res['drug']} | Interaction: {res['interaction_type']} | Score: {res['score']}, Edge Type: {res['edge_type']}")