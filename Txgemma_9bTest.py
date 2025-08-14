#!/usr/bin/env python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── 1. Setup & load model in FP16 on GPU / FP32 on CPU ─────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model_name = "google/txgemma-9b-chat"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

# Load in half‑precision on GPU, full precision on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                 # let Accelerate shard across GPU/CPU
    torch_dtype=(torch.float16 if device.type=="cuda" else torch.float32),
    output_hidden_states=True,
    return_dict_in_generate=True
)
model.eval()
print("TxGemma loaded (FP16 on GPU / FP32 on CPU, hidden states enabled).\n")


# ─── 2. Small drug database ─────────────────────────────────────────────────────

known_drugs = {
    "Aspirin": {
        "smiles":    "CC(=O)OC1=CC=CC=C1C(=O)O",
        "mechanism": "Irreversible COX‑1/2 inhibitor; reduces prostaglandins"
    },
    "Ibuprofen": {
        "smiles":    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "mechanism": "Reversible COX‑1/2 inhibitor; anti‑inflammatory"
    },
    "Paracetamol": {
        "smiles":    "CC(=O)NC1=CC=C(C=C1)O",
        "mechanism": "Likely COX‑3 inhibition in CNS; analgesic/antipyretic"
    },
    "Metformin": {
        "smiles":    "CNC(N)=N/C(=N/C)N",
        "mechanism": "Activates AMPK; lowers hepatic gluconeogenesis"
    },
    "Simvastatin": {
        "smiles":    "CC[C@H](C)[C@H]1CCC2C1(CCC3C2CCC(=O)O3)OC",
        "mechanism": "HMG‑CoA reductase inhibitor; lowers LDL cholesterol"
    },
    "Lisinopril": {
        "smiles":    "C[C@H](N)C[C@@H](C(=O)O)N1CCCC1C(=O)O",
        "mechanism": "ACE inhibitor; lowers blood pressure"
    }
}


# ─── 3. Embedding helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def embed_smiles(smi: str):
    # tokenize *without* moving to cuda
    tokens = tokenizer(smi, return_tensors="pt")
    outputs = model(**tokens)
    # mean‑pool last hidden layer
    return outputs.hidden_states[-1].mean(dim=1)  # (1, hidden_dim)


# ─── 4. Similarity search ────────────────────────────────────────────────────────

def find_top_k(query_smi, k=5):
    q_emb = embed_smiles(query_smi)
    scores = {}
    for name, info in known_drugs.items():
        db_emb      = embed_smiles(info["smiles"])
        scores[name] = F.cosine_similarity(q_emb, db_emb, dim=1).item()
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# ─── 5. Build explanation prompt ─────────────────────────────────────────────────

def build_explain_prompt(query_smi, hits):
    lines = []
    for i, (name, score) in enumerate(hits, 1):
        info = known_drugs[name]
        lines.append(
            f"{i}) {name}\n"
            f"   SMILES:    {info['smiles']}\n"
            f"   Mechanism: {info['mechanism']}\n"
            f"   Similarity: {score:.3f}\n"
        )
    block = "\n".join(lines)
    return (
        "You are TxGemma, an expert in drug similarity.\n\n"
        f"Query SMILES: {query_smi}\n\n"
        "Here are the top 5 most similar known drugs:\n\n"
        f"{block}\n"
        "For each, **explain why** it is biologically similar to the query—"
        "mention shared structural motifs, targets, or pathways.\n"
        "Answer as:\n"
        "Drug Name – Explanation of similarity\n"
    )


# ─── 6. Main interactive flow ────────────────────────────────────────────────────

def main():
    txt = input("Enter a known drug name (e.g. Aspirin) or paste any SMILES:\n> ").strip()
    if txt in known_drugs:
        name, query_smi = txt, known_drugs[txt]["smiles"]
        print(f"\n🔷 Query: {name} – {query_smi}\n")
    else:
        name, query_smi = None, txt
        print(f"\n🔷 Query SMILES: {query_smi}\n")

    print("🔍 Computing similarities…")
    top_hits = find_top_k(query_smi)

    print("\nTop 5 candidates:")
    for nm, sc in top_hits:
        print(f" • {nm} ({known_drugs[nm]['smiles']}): {sc:.3f}")

    print("\n✏️ Asking TxGemma to explain…\n")
    # tokenize *without* .to(device)
    prompt = build_explain_prompt(query_smi, top_hits)
    tokens = tokenizer(prompt, return_tensors="pt")
    out    = model.generate(
        **tokens,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    explanation = tokenizer.decode(out[0], skip_special_tokens=True)
    print(explanation)


if __name__ == "__main__":
    main()
