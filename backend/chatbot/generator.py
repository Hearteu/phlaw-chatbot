import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/law-chat")
model = AutoModelForCausalLM.from_pretrained("AdaptLLM/law-chat")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt):
    # Tokenize with truncation and move to correct device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    # Generate with sampling, optional temperature and top_p for more variety
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id  # prevent warning
    )

    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Optionally remove the prompt from the output if it echoes it
    return response.replace(prompt, "").strip()
