"""
Ex. No: 8
IMAGE GENERATION APPLICATION USING DIFFUSION MODELS

Note: runs on a 4GB VRAM GPU (RTX 3050) using fp16 weights + attention slicing
(memory optimisation not in the original manual snippet, needed for this hardware).
Falls back to CPU automatically if no CUDA device is available.
"""
from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated_city.png")
print("Image generated and saved as generated_city.png")
