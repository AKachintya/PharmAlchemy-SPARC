# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:45:04 2025

@author: Akach
"""

from txgemma import TxGemmaClient  # Gemma 2 LLM for therapeutics :contentReference[oaicite:10]{index=10}
import pandas as pd

# Load drug metadata with descriptions
drugs = pd.read_pickle("data/drugs.pkl")  # must have 'name' and 'description' columns

client = TxGemmaClient(model="txgemma-2b-predict")  # choose a model :contentReference[oaicite:11]{index=11}

def embed_drug_text(name, desc):
    """Return a semantic embedding for a drug given its description."""
    prompt = f"{name}: {desc}"
    emb = client.embed_text(prompt)
    return emb

# Compute embeddings for all drugs
semantic_embs = []
for _, row in drugs.iterrows():
    emb = embed_drug_text(row['name'], row['description'])
    semantic_embs.append(emb)

# Convert to array and save
semantic_embs = pd.DataFrame(semantic_embs, index=drugs.index)
semantic_embs.to_pickle("output/semantic_embs.pkl")
