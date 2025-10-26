import torch

# Use MPS if available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Example for diffusers pipeline
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a futuristic city with palm trees and sky bridges, ultra-detailed, 4k"
image = pipe(prompt).images[0]
image.save("output.png")
print("✅ Image saved with GPU acceleration!")
