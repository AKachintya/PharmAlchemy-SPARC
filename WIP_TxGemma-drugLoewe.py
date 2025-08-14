import torch
import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def run_loewe_prediction():
    """
    This script uses the txgemma-9b-predict model to predict Loewe synergy scores
    for drug combinations from the TDC DrugComb_Loewe benchmark.
    """
    print("--- 1. Initializing txgemma-9b-predict model ---")
    
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("GPU detected. Using CUDA for inference.")
    else:
        print("No GPU detected. Using CPU for inference (this may be slow).")

    model_name = "google/txgemma-9b-predict"

    # ✅ --- The Fix: Load model and tokenizer separately for more control --- ✅
    
    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load the model, explicitly setting the Windows-compatible attention implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # This forces a compatible backend
    )

    # 3. Create the pipeline using the pre-loaded components
    predictor = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Apply the gemma chat template to the tokenizer
    predictor.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{% if loop.index0 == 0 %}{{ '' }}{% else %}{{ '<end_of_turn>\\n' }}{% endif %}{{ 'user\\n' + message['content'] }}{% elif (message['role'] == 'assistant') %}{{ '<end_of_turn>\\n' + 'model\\n' + message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<end_of_turn>\\nmodel\\n' }}{% endif %}"


    print("\n--- 2. Loading DrugComb_Loewe Benchmark Dataset ---")

    base_path = r"C:\Users\Akach\drugcomb_loewe"
    test_file_path = os.path.join(base_path, "test.pkl")

    test_df = pd.read_pickle(test_file_path)
    
    print(f"Successfully loaded the dataset from {test_file_path}")
    print(f"Number of test samples: {len(test_df)}")
    print(f"Dataset columns: {test_df.columns.tolist()}")

    sample_size = 5
    test_sample = test_df.sample(n=sample_size, random_state=42)
    
    print(f"\n--- 3. Running Predictions on {sample_size} Test Samples ---")

    for index, row in test_sample.iterrows():
        drug1_smiles = row['Drug1']
        drug2_smiles = row['Drug2']
        cell_line = row['CellLine']
        true_loewe_score = row['Y']

        prompt = (
            f"Predict the Loewe score for drug1: {drug1_smiles}, "
            f"drug2: {drug2_smiles} on cell line: {cell_line}"
        )
        
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            output = predictor(
                messages,
                max_new_tokens=10, 
                do_sample=False,
            )
            
            predicted_text = output[0]['generated_text'].split('model\n')[-1].strip()
            predicted_loewe_score = float(predicted_text)

            print("\n-------------------------------------------")
            print(f"Sample with Index: {index}")
            print(f"  Drug 1 (SMILES): {drug1_smiles[:40]}...")
            print(f"  Drug 2 (SMILES): {drug2_smiles[:40]}...")
            print(f"  Cell Line: {cell_line}")
            print(f"  ✅ True Loewe Score:    {true_loewe_score:.4f}")
            print(f"  🤖 Predicted Loewe Score: {predicted_loewe_score:.4f}")
            print("-------------------------------------------")

        except (ValueError, IndexError, UnboundLocalError) as e:
            print(f"\nCould not parse prediction for sample #{index}. Error: {e}")

if __name__ == "__main__":
    run_loewe_prediction()