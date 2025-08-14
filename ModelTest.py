import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

### Load Model and Tokenizer ###
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

### Compute Embeddings ###
def compute_embeddings(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

### Process JSON Data ###
def process_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    dataset_texts, metadata_list = [], []
    for edge in data["hyperedges"].values():
        gene = edge["hyperedge_id"]
        for drug, info in edge["paired_drugs"].items():
            itype = info["interaction_type"] if info["interaction_type"] else "unknown"
            score = info.get("r2g_score", 0.0)
            text = f"Gene: {gene} | Drug: {drug} | Interaction: {itype} | Score: {score}"
            dataset_texts.append(text)
            metadata_list.append({"gene": gene, "drug": drug, "interaction_type": itype, "score": score})
    return dataset_texts, metadata_list

### Filter Results Based on Metadata ###
def filter_results_by_metadata(query, retrieved_metadata):
    query_tokens = query.lower().split()
    interaction_type = next((token for token in ["agonist", "inhibitor", "modulator"] if token in query_tokens), None)
    # Allow partial matches for interaction types
    return [r for r in retrieved_metadata if interaction_type is None or interaction_type in r["interaction_type"].lower()]

### Evaluate Model on Query ###
def evaluate_model_on_query(model_name, dataset_texts, metadata_list, query, k=5):
    print(f"Evaluating model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset_embeddings = compute_embeddings(dataset_texts, model, tokenizer)
    query_embedding = compute_embeddings([query], model, tokenizer)
    sims = cosine_similarity(query_embedding, dataset_embeddings)[0]
    top_indices = np.argsort(-sims)[:k]
    top_metadata = [metadata_list[i] for i in top_indices]
    filtered_results = filter_results_by_metadata(query, top_metadata)
    return model_name, query, filtered_results, sims[top_indices]  # Return similarity scores for debugging

### Main Evaluation ###
if __name__ == "__main__":
    file_path = "C:/Users/Akach/Downloads/AlzheimersR2Ghypergraph.json"
    dataset_texts, metadata_list = process_json_file(file_path)
    queries = ["agonist drugs for APP gene", "inhibitor drugs for ADRB1 gene"]
    models = ["allenai/biomed_roberta_base", "dmis-lab/biobert-v1.1", "medicalai/ClinicalBERT"]
    results = []
    for model_name in models:
        for query in queries:
            res = evaluate_model_on_query(model_name, dataset_texts, metadata_list, query, k=5)
            results.append(res)
    for r in results:
        print(f"Model: {r[0]}")
        print(f"Query: {r[1]}")
        print("Filtered Results:")
        if not r[2]:  # If no results, print a message
            print("  No results found.")
        else:
            for item, score in zip(r[2], r[3]):  # Include similarity score in output
                print(f"  Gene: {item['gene']} | Drug: {item['drug']} | Interaction: {item['interaction_type']} | Score: {item['score']} | Similarity: {score:.4f}")
        print()