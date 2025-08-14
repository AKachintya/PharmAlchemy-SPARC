from flask import Flask, request, jsonify
from transformers import pipeline
from sqlalchemy import create_engine, text

# Detailed Instructions:
# 1. Replace the placeholders (DB_USER, DB_PASSWORD, etc.) with your database credentials or use environment variables for security.
# 2. Ensure you have the required Python packages installed: flask, transformers, sqlalchemy, and cx_oracle.
# 3. Start the Flask server and use the endpoints for semantic search and database querying.

# Load the BioBERT model from Hugging Face
# Model: BioBERT (dmis-lab/biobert-base-cased-v1.1)
# This model is designed for biomedical text processing, particularly for question-answering tasks.
print("Loading BioBERT model...")
qa_pipeline = pipeline("question-answering", model="dmis-lab/biobert-base-cased-v1.1")
print("BioBERT model loaded.")

# Flask app initialization
app = Flask(__name__)

# Oracle database connection setup
DB_USER = "drg_depot_staging"
DB_PASSWORD = "drgdb"
DB_HOST = "infoinst-02.rc.uab.edu"
DB_PORT = "1521"
DB_SERVICE = "BIODB.RC.UAB.EDU"

DATABASE_URL = f"oracle+cx_oracle://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?service_name={DB_SERVICE}"

# Create SQLAlchemy engine for database connection
try:
    engine = create_engine(DATABASE_URL)
    print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {e}")

@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    """
    Endpoint for semantic search using BioBERT.
    Input JSON:
      {
        "question": "What is the treatment for hypertension?",
        "context": "Hypertension is treated with lifestyle changes and medications."
      }
    Response JSON:
      {
        "score": 0.95,
        "start": 0,
        "end": 48,
        "answer": "lifestyle changes and medications"
      }
    """
    try:
        # Parse JSON request
        data = request.get_json()
        question = data.get("question")
        context = data.get("context")

        if not question or not context:
            return jsonify({"error": "Both 'question' and 'context' fields are required."}), 400

        # Use BioBERT for semantic search
        result = qa_pipeline({"question": question, "context": context})
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query-database", methods=["POST"])
def query_database():
    """
    Endpoint to execute SQL queries on the Oracle database.
    Input JSON:
      {
        "query": "SELECT * FROM your_table_name"
      }
    Response JSON:
      [
        {"column1": "value1", "column2": "value2"},
        {"column1": "value3", "column2": "value4"}
      ]
    """
    try:
        # Parse JSON request
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "The 'query' field is required."}), 400

        # Execute query using SQLAlchemy
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = [dict(row) for row in result]

        return jsonify(rows)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run Flask app on localhost
    # Access it via http://127.0.0.1:5000 or http://localhost:5000
    app.run(host="0.0.0.0", port=5000)
