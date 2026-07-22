"""
Ex. No: 3
CONVERSATIONAL AI CHATBOT USING TRANSFORMER-BASED LANGUAGE MODELS

Note: the lab manual's original script reads user turns interactively via input().
For unattended, reproducible execution and logging, this version drives the same
DialoGPT multi-turn logic with a fixed list of sample user turns instead of input().
The core algorithm (history concatenation, generation, decoding) is unchanged.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

sample_user_turns = [
    "Hi, how are you?",
    "What can you help me with?",
    "Tell me something interesting about AI.",
]

print("Chatbot ready! Running sample multi-turn conversation.")
for step, user_input in enumerate(sample_user_turns):
    print(f">> User: {user_input}")

    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")
