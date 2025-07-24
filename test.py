import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose model name
model_name = "AdaptLLM/law-chat"

# Load model (auto device map for mixed GPU/CPU offload)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# System prompt
system_prompt = """
You are a helpful, respectful, and honest legal assistant.
Your answers should be safe, unbiased, and fact-based.

If a question is unclear or incoherent, kindly explain why.
If unsure, say "I don't know" instead of guessing.
"""

# Response generation function
def generate_response(message, chat_history):
    # Format previous messages
    history = "\n".join([f"User: {user}\nAssistant: {bot}" for user, bot in chat_history])
    user_prompt = f"{history}\nUser: {message}\nAssistant:"

    # Full prompt using LLaMA2-style format
    full_prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n{user_prompt} [/INST]"

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,             # Faster generation
            do_sample=False,                # Deterministic
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    answer_start = inputs['input_ids'].shape[-1]
    response = tokenizer.decode(outputs[0][answer_start:], skip_special_tokens=True)

    # Append to chat history
    chat_history.append((message, response.strip()))
    return "", chat_history, chat_history

# Gradio UI
with gr.Blocks(title="Legal Chatbot") as demo:
    gr.Markdown("## ⚖️ Philippine Legal Chatbot (AdaptLLM)")
    chatbot = gr.Chatbot(label="Law Chat")
    msg = gr.Textbox(label="Enter your legal question here", placeholder="Type your question and press Enter...")
    clear = gr.Button("Clear")

    state = gr.State([])  # Chat history

    msg.submit(generate_response, [msg, state], [msg, chatbot, state])
    clear.click(lambda: ([], [], []), outputs=[msg, chatbot, state])

# Launch app
demo.launch()
