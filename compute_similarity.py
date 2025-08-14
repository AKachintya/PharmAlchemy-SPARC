# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:44:29 2025

@author: Akach
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs  # RDKit fingerprint & similarity :contentReference[oaicite:8]{index=8}

# --- Load pre-saved dataframes ---
drugs = pd.read_pickle("data/drugs.pkl")  # must have 'smiles' column
lincs = pd.read_pickle("data/lincs.pkl")  # index = drug_id, columns = genes
cp = pd.read_pickle("data/cell_painting.pkl")  # index = drug_id, columns = features

def compute_fingerprints(smiles_list):
    """Convert SMILES to Morgan fingerprints."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(fp)
    return fps

# 1. Chemical similarity (Tanimoto)
fps = compute_fingerprints(drugs['smiles'])
n = len(fps)
chem_sim = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        score = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        chem_sim[i, j] = chem_sim[j, i] = score

# 2. Transcriptomic similarity (Pearson)
#    assumes lincs rows align with drugs order
trans_sim = lincs.corr(method="pearson").values

# 3. Morphological similarity (Spearman)
morph_sim = cp.corr(method="spearman").values

# Save similarity matrices
np.save("output/chem_sim.npy", chem_sim)
np.save("output/transcriptomic_sim.npy", trans_sim)
np.save("output/morphological_sim.npy", morph_sim)
