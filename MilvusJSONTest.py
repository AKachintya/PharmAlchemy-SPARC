import torch
import pandas as pd
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 10000
MODEL_NAME = "medicalai/ClinicalBERT"

### Load Model and Tokenizer ###
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def process_csv(file_path):
    df = pd.read_csv(file_path)
    required_cols = {"hyperedge_id", "drug_name", "interaction_type", "r2g_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file missing required columns: {required_cols - set(df.columns)}")
    
    texts = [f"{row['hyperedge_id']} - {row['drug_name']} ({row['interaction_type']}), Score: {row['r2g_score']}" 
             for _, row in df.iterrows()]
    return texts, generate_embeddings(texts)

def process_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    texts = []
    
    for edge in data["hyperedges"].values():
        gene = edge["hyperedge_id"]
        for drug, info in edge["paired_drugs"].items():
            interaction = info.get("interaction_type", "unknown")
            score = info.get("r2g_score", 0.0)
            texts.append(f"{gene} - {drug} ({interaction}), Score: {score}")
    return texts, generate_embeddings(texts)

def create_milvus_collection(collection_name):
    """Creates a Milvus collection if it doesn't exist."""
    connections.connect("default", host="localhost", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary ID
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Text
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # Embeddings
    ]
    
    schema = CollectionSchema(fields, description="Gene-drug interaction embeddings")

    if collection_name not in utility.list_collections():
        collection = Collection(name=collection_name, schema=schema)
        print(f"Created collection: {collection_name}")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    
    return collection

def store_embeddings_milvus(collection_name="gene_drug_embeddings", file_path="data.json", file_type="json"):
    """Store embeddings in Milvus while handling batch insertions."""
    connections.connect("default", host="localhost", port="19530")
    
    collection = create_milvus_collection(collection_name)  # Ensure collection exists

    # Process file based on type (JSON or CSV)
    if file_type == "json":
        texts, embeddings = process_json(file_path)
    else:
        texts, embeddings = process_csv(file_path)
    
    num_records = len(texts)
    print(f"Processing {num_records} records...")

    for i in range(0, num_records, BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]

        # Insert data in column format (list per field)
        # **Remove the manual ID generation since auto_id=True**
        data = [
            batch_texts,  # Text field
            batch_embeddings.tolist()  # Embedding field
        ]

        collection.insert(data)  # Correct format
        print(f"Inserted batch {i // BATCH_SIZE + 1}/{(num_records // BATCH_SIZE) + 1}")

    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
    collection.load()

def search_milvus(query_text, collection_name, top_k=5):
    collection = Collection(collection_name)
    query_embedding = generate_embeddings([query_text])[0]
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", search_params, limit=top_k, output_fields=["text"])
    return [(hit.entity.get("text"), hit.distance) for hit in results[0]]

if __name__ == "__main__":
    store_embeddings_milvus("gene_drug_embeddings", "C:/Users/Akach/Downloads/AlzheimersUnifiedHetNet", "json")
    
    query = "drugs that increase the biological activity of the APP gene"
    results = search_milvus(query, "gene_drug_embeddings")
    print("Search results:", results)