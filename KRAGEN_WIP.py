import torch
import pandas as pd
import json
from weaviate import WeaviateClient
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "medicalai/ClinicalBERT"
WEAVIATE_CLASS_NAME = "KRAGENDrugGeneInteractionNoBatchV4" # Updated class name to indicate v4 API and no batching

# Initialize Weaviate client
import weaviate

client = weaviate.connect_to_local()  # Connect with default parameters

# --- Load ClinicalBERT Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# --- Function to Generate Embeddings ---
def generate_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# --- Function to Process JSON File ---
def process_json(file_path):
    """
    Parses JSON, extracts relevant data, and generates embeddings.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    data_objects = []

    for edge in data["hyperedges"].values():  # Iterating through hyperedges
        gene = edge["hyperedge_id"]  # Gene ID
        for drug, info in edge["paired_drugs"].items():  # Paired drugs
            interaction = info.get("interaction_type", "unknown")
            score = info.get("r2g_score", 0.0)

            text_description = f"{gene} - {drug} ({interaction}), Score: {score}"
            embedding = generate_embedding(text_description)

            data_object = {
                "gene_name": gene,
                "drug_name": drug,
                "interaction_type": interaction,
                "r2g_score": score,
                "text_description": text_description,
                "embedding": embedding.tolist()  # Convert NumPy array to list for JSON storage
            }
            data_objects.append(data_object)

    return data_objects

# --- Function to Store Data in Weaviate ---
def store_in_weaviate(file_path):
    """
    Stores processed embeddings into Weaviate.
    """
    data_objects = process_json(file_path)
    collection = client.collections.get(WEAVIATE_CLASS_NAME)  # Ensure the class exists in Weaviate

    for obj in data_objects:
        collection.data.insert(properties=obj)

    print(f"Stored {len(data_objects)} records in Weaviate.")

# --- Run the Processing & Storage ---
if __name__ == "__main__":
    file_path = "C:/Users/Akach/Downloads/AlzheimersR2Ghypergraph.json"  # Update path as needed
    store_in_weaviate(file_path)