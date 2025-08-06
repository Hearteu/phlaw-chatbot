# import torch

# torch.set_num_threads(1)  # Prevent CPU contention

# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/law-chat")
# model = AutoModelForCausalLM.from_pretrained(
#     "AdaptLLM/law-chat",
#     torch_dtype=torch.float16  # if you have GPU!
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def generate_response(prompt, max_new_tokens=64):
#     prompt = prompt[:1500]
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,  # for speed/determinism
#             pad_token_id=tokenizer.eos_token_id
#         )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.replace(prompt, "").strip()


import os

from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "law-chat.Q5_K_M.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=8
)

def generate_response(prompt):
    output = llm(
        prompt,
        max_tokens=3000,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "Sources:", "Question:", "\n\n"]
    )
    return output['choices'][0]['text'].strip()
