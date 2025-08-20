import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Hugging Face authentication (token needs to be set in environment)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Base model (already in your RunPod volume)
MODEL_PATH = "defog/sqlcoder-7b-2"

# LoRA weights from your Hugging Face private repo
LORA_REPO = "Rohit1993/sqlcoder-lora"

print("Loading SQLCoder model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=HF_TOKEN)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=HF_TOKEN
)

# Apply LoRA
model = PeftModel.from_pretrained(base_model, LORA_REPO, use_auth_token=HF_TOKEN)

print("Base model + LoRA weights loaded successfully.")

ENRICHED_SCHEMA = """
Table: inventory_snapshots
Columns:
- id (bigserial, primary key) → unique snapshot ID
- item_unique_id (text, not null) → unique identifier for each item
- class_title (text) → category/class of the item
- approx (text) → approximate quantity or measurement
- description (text) → item description. Supports:
    • Fuzzy search with trigram similarity: description % 'term'
    • Similarity scoring: similarity(description, 'term') > 0.1
    • Partial match: description ILIKE '%term%'
- prediction_date (timestamptz) → timestamp of the snapshot
- embedding (vector[1536]) → semantic embedding for similarity search.
    • To find semantically similar items, order by embedding <-> '[vector]'
- metadata (jsonb) → raw JSON snapshot (queryable with JSON operators)

Indexes available:
- Trigram search index on description
- ivfflat index on embedding (cosine similarity)
- GIN index on metadata

Instructions:
- Use trigram (%) or similarity() for fuzzy search on description
- Use embedding <-> '[vector]' for semantic search
- Use JSON operators for metadata
- Combine these approaches as appropriate depending on the question
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Q: Find all items with descriptions similar to 'blood pressure monitor'.
A: SELECT * FROM inventory_snapshots WHERE description % 'blood pressure monitor';

Example 2:
Q: Retrieve the top 5 items semantically similar to 'oxygen concentrator'.
A: SELECT * FROM inventory_snapshots ORDER BY embedding <-> '[vector]' LIMIT 1000;

Example 3:
Q: Find items in class 'Medical Devices' with description like 'ventilator'.
A: SELECT * FROM inventory_snapshots WHERE class_title = 'Medical Devices' AND description % 'ventilator';
"""

def handler(event):
    input_data = event.get('input', {})
    question = input_data.get('question')
    temperature = input_data.get('temperature', 0.2)  # default to 0.2 if not provided

    if not question:
        return {"error": "Please provide a 'question' in the input."}

    # Build final prompt
    final_prompt = FEW_SHOT_EXAMPLES + "\n\nSchema:\n" + ENRICHED_SCHEMA + f"\n\nQ: {question}\nA:"

    print("Prompt to model:\n", final_prompt)
    print(f"Using temperature: {temperature}")

    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        do_sample=True,
        temperature=float(temperature),  # use the provided temperature
        top_k=50,
        top_p=0.95
    )
    sql_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"sql_query": sql_output.strip()}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
