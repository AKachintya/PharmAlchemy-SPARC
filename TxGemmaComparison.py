import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load tdc_prompts.json from Hugging Face
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-27b-chat",
    filename="tdc_prompts.json"
)

with open(tdc_prompts_filepath, "r") as f:
    tdc_prompts_json = json.load(f)

# Step 2: Choose 5 target tasks and valid SMILES inputs
tasks_with_inputs = {
    "BBB_Martins": "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21",         # Clozapine
    "Caco2_Wang": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                  # Ibuprofen
    "CYP2D6_Veith": "CC(C)NCC(O)c1ccc(O)cc1",                       # Propranolol
    "Pgp_Broccatelli": "CCOC(=O)C1=CC=CC=C1Cl",                     # Verapamil
    "Bioavailability_Ma": "CCOC(=O)C1=CC=CC=C1Cl",                  # Verapamil
}

# Step 3: Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/txgemma-27b-chat")
model = AutoModelForCausalLM.from_pretrained("google/txgemma-27b-chat", device_map="auto")

# Step 4: Loop through each task and generate output
for task_name, drug_smiles in tasks_with_inputs.items():
    print("=" * 60)
    print(f"🔬 Task: {task_name}")

    prompt_template = tdc_prompts_json[task_name]

    # Replace placeholder or edit prompt with real input
    if "{Drug SMILES}" in prompt_template:
        prompt = prompt_template.replace("{Drug SMILES}", drug_smiles)
    else:
        # If placeholder not found, try inserting manually
        if "Drug SMILES:" in prompt_template:
            prompt = prompt_template.split("Output:")[0].strip() + f"\nOutput:"
        else:
            prompt = prompt_template
        prompt = prompt.replace("Drug SMILES:", f"Drug SMILES: {drug_smiles}")

    # Make sure only prompt and no pre-filled output is passed
    if "Output:" in prompt:
        prompt = prompt.split("Output:")[0].strip() + "\nOutput:"

    print(f"🧪 Prompt:\n{prompt}")

    # Tokenize and query model
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=8)

    # Decode and print output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Model Prediction:\n{output_text.strip()}")
