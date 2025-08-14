# test_drugcombo_offload.py

import os
import json
import re
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from tdc.benchmark_group import drugcombo_group

# 1) Prepare offload directory
offload_dir = "offload_dir"
os.makedirs(offload_dir, exist_ok=True)

# 2) Download & load the TDC prompt templates
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-chat",
    filename="tdc_prompts.json",
)
with open(tdc_prompts_filepath, "r") as f:
    prompts = json.load(f)

# 3) Define your test combination
drug1_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"         # Ibuprofen
drug2_smiles = "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21"  # Clozapine
cell_line    = "A2780"

# 4) Load TxGemma‑9B‑Chat with disk offload
model_id  = "google/txgemma-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    offload_folder=offload_dir,
    offload_state_dict=True
)
device = model.device

# 5) Helper: generate numeric prediction from TxGemma prompt
def predict(task_key: str) -> float:
    tpl = prompts[task_key]
    prompt = (
        tpl
        .replace("{Drug1 SMILES}", drug1_smiles)
        .replace("{Drug2 SMILES}", drug2_smiles)
        .replace("{Cell line description}", cell_line)
    )
    # Force the model to fill after "Answer:"
    if "Answer:" in prompt:
        prompt = prompt.split("Answer:")[0].strip() + "\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=12)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
    return float(m.group()) if m else None

# 6) Get TxGemma predictions
pred_css = predict("DrugComb_CSS")
pred_hsa = predict("DrugComb_HSA")

# 7) Load & concatenate TDC splits into DataFrames
group   = drugcombo_group(path="data/")
css_splits = group.get("DrugComb_CSS")
css_df     = pd.concat([df for df in css_splits.values() if isinstance(df, pd.DataFrame)], ignore_index=True)
hsa_splits = group.get("DrugComb_HSA")
hsa_df     = pd.concat([df for df in hsa_splits.values() if isinstance(df, pd.DataFrame)], ignore_index=True)

# 8) Lookup function using known column names
def lookup(df, col):
    sub = df[
        ((df["Drug1"] == drug1_smiles) & (df["Drug2"] == drug2_smiles) & (df["CellLine"] == cell_line)) |
        ((df["Drug1"] == drug2_smiles) & (df["Drug2"] == drug1_smiles) & (df["CellLine"] == cell_line))
    ]
    return float(sub.iloc[0][col]) if not sub.empty else None

# 9) True values are stored in column "Y"
true_css = lookup(css_df, "Y")
true_hsa = lookup(hsa_df, "Y")

# 10) Report
print(f"TxGemma predicted CSS = {pred_css}; True CSS = {true_css}")
print(f"TxGemma predicted HSA = {pred_hsa}; True HSA = {true_hsa}")
