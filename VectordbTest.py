# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:55:48 2025
@author: Akach
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sqlalchemy import create_engine, text
import torch
import os
import uuid  # For generating valid Qdrant IDs

app = Flask(__name__)

# Load BioMed-RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

def embed_text(text):
    """Generate embedding for the given text using BioMed-RoBERTa."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Convert to list for Qdrant
    return embedding

# Qdrant client initialization
qdrant_client = QdrantClient(host='127.0.0.1', port=6333)

# Define collection name
collection_name = "pharmalchemy_embeddings"

# Check if the collection exists, create it if missing
if not qdrant_client.collection_exists(collection_name=collection_name):
    print(f"🛠️ Creating Qdrant collection: {collection_name}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# Oracle database connection setup
DB_USER = os.getenv("DB_USER", "drg_depot_staging")
DB_PASSWORD = os.getenv("DB_PASSWORD", "drgdb")
DB_HOST = os.getenv("DB_HOST", "infoinst-02.rc.uab.edu")
DB_PORT = os.getenv("DB_PORT", "1521")
DB_SERVICE = os.getenv("DB_SERVICE", "BIODB.RC.UAB.EDU")

DATABASE_URL = f"oracle+cx_oracle://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?service_name={DB_SERVICE}"

# Create SQLAlchemy engine for database connection
engine = create_engine(DATABASE_URL)

@app.route("/generate-embeddings", methods=["POST"])
def generate_embeddings():
    """
    Generate and store embeddings from database text fields.
    """
    try:
        print("🔍 Received /generate-embeddings request")  # Debugging
        data = request.get_json()
        query = data.get("query")

        if not query:
            print("❌ No query provided")
            return jsonify({"error": "The 'query' field is required."}), 400

        print(f"🟢 Running query: {query}")

        # Execute query to fetch data from Oracle DB
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = [dict(zip(result.keys(), row)) for row in result]

        print(f"✅ Fetched {len(rows)} rows from database.")

        if not rows:
            return jsonify({"error": "No data found for the provided query."}), 404

        # Debug: Print available columns
        print(f"📝 Available columns: {rows[0].keys()}")

        # Generate and store embeddings in Qdrant
        points = []
        for row in rows:
            text_column = row.get("description")  # Replace with the correct column name
            if text_column:
                embedding = embed_text(text_column)
                point_id = str(uuid.uuid4())  # Use UUID for unique Qdrant IDs
                print(f"💾 Storing embedding: ID={point_id}, Text={text_column[:30]}..., Vector Size={len(embedding)}")
                points.append(PointStruct(id=point_id, vector=embedding, payload={"text": text_column}))

        if points:
            qdrant_client.upsert(collection_name=collection_name, points=points)

        return jsonify({"result": "Embeddings generated and stored successfully.", "count": len(points)})

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    """
    Perform a semantic search using Qdrant.
    """
    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "The 'query' field is required."}), 400

        # Generate embedding for the query
        query_embedding = embed_text(query)

        # Perform search in Qdrant
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5  # Number of results to return
        )

        # Format the response
        response = [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", "No text found")
            }
            for result in search_result
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test-qdrant", methods=["GET"])
def test_qdrant():
    """
    Insert a test embedding into Qdrant.
    """
    from qdrant_client.models import PointStruct
    import numpy as np

    # Sample test embedding
    text_sample = "Lung cancer is a serious disease."
    test_embedding = np.random.rand(768).tolist()

    # Generate a valid UUID for the point ID
    valid_point_id = str(uuid.uuid4())

    print(f"📝 Inserting test embedding with ID: {valid_point_id}")

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=valid_point_id,  # Use UUID instead of "test123"
                vector=test_embedding,
                payload={"text": text_sample}
            )
        ]
    )

    count = qdrant_client.count(collection_name=collection_name)
    return jsonify({"result": "Test embedding inserted", "total_embeddings": count.count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
