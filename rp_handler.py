import runpod
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model + tokenizer at startup (only once per worker)
MODEL_NAME = "defog/sqlcoder-7b-2"
print("Loading model... this may take a while on cold start.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)
print("Model loaded.")

def handler(event):
    print("Worker Start")
    input = event['input']

    # Accept either (prompt) OR (question+schema)
    prompt = input.get('prompt')
    question = input.get('question')
    schema = input.get('schema', '')

    if not prompt and not question:
        return {"error": "Must provide either 'prompt' or both 'question' and 'schema'."}

    if question:
        # SQLCoder works best with schema + question
        prompt = f"-- Schema:\n{schema}\n-- Question:\n{question}\n-- SQL:\n"

    print(f"Received prompt:\n{prompt}")

    # Generate SQL
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=256,
        do_sample=False
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"sql": sql}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
