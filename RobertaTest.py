# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:50:34 2025

@author: Akach
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

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
    
    return embeddings

def process_csv(file_path):
    """
    Read CSV file and process text data for embedding generation.
    """
    df = pd.read_csv(file_path)
    
    # Identify a relevant text column
    possible_text_cols = ["text", "Disease Name"]
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    
    if text_col is None:
        raise ValueError(f"CSV file must contain a text-related column. Found columns: {df.columns}")
    
    return generate_embeddings(df[text_col].astype(str).tolist())

# Example test
if __name__ == "__main__":
    csv_file = "C:/Users/Akach/Downloads/DISEASES.csv"
    embeddings = process_csv(csv_file)
    print("Embeddings shape:", embeddings.shape)
    print("Sample embedding:", embeddings[0])
