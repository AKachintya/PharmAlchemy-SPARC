# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:47:27 2025

@author: Akach
"""

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load data once at startup
drugs = pd.read_pickle("data/drugs.pkl")
chem_sim = np.load("output/chem_sim.npy")

@app.route("/similarity/chemical")
def get_chemical_similarity():
    """
    Query: ?drug_id=123&top_k=10
    Returns top-K chemicals most similar to the given drug.
    """
    drug_id = int(request.args.get("drug_id"))  # index in drugs
    top_k = int(request.args.get("top_k", 10))
    sims = chem_sim[drug_id]
    inds = sims.argsort()[::-1][1: top_k+1]  # exclude self (index 0)
    results = [{"drug_id": int(i), "score": float(sims[i])} for i in inds]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
