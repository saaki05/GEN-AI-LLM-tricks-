"""
Ex. No: 1
TEXT GENERATION USING PRE-TRAINED FOUNDATION MODELS
"""
from transformers import pipeline, set_seed

# Load the pre-trained GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Input prompt
prompt = "Artificial Intelligence will transform the future of"

# Generate text
outputs = generator(
    prompt,
    max_length=60,
    max_new_tokens=None,  # transformers>=4.50 defaults max_new_tokens=256, which
                          # silently overrides max_length; set to None to honour
                          # the manual's max_length parameter as intended.
    num_return_sequences=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

for i, out in enumerate(outputs, 1):
    print(f"--- Generated Text {i} ---")
    print(out["generated_text"])
    print()
