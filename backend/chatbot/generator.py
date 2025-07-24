import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/law-chat")
model = AutoModelForCausalLM.from_pretrained("AdaptLLM/law-chat")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_length=1024, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
