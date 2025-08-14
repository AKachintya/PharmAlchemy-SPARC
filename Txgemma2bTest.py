# -*- coding: utf-8 -*-
"""
Inference in FP16 with CPU-offload to fit 2B-param model on 12 GB GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Clear any leftover memory
torch.cuda.empty_cache()

# Select device for offload
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "google/txgemma-2b-predict"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in FP16 with CPU offload
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",            # auto-offload layers to CPU/GPU
    torch_dtype=torch.float16,    # half-precision
    low_cpu_mem_usage=True        # reduce peak CPU memory
)
model.eval()

print("Model loaded in FP16 with CPU offload.")

# Example inference
drug_smiles = "COC1=CC=CC=C1C(=O)NC2=CC=CC=C2"
prompt = f"""Instructions: Predict whether the compound is active or inactive against HIV replication.
Question: (A) Inactive  (B) Active
Drug SMILES: {drug_smiles}
Answer:"""

# Tokenize and move inputs to correct device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=8)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Prediction:", response)
