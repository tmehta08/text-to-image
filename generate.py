import torch
from diffusers import StableDiffusionPipeline
from loguru import logger
import time
import sys

# ----- 1. Configure logger -----
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("generation.log", rotation="10 MB")
logger.info("Starting Stable Diffusion pipeline")

# ----- 2. Device -----
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Using device: {device}")

# ----- 3. Load model -----
model_id = "runwayml/stable-diffusion-v1-5"
torch_dtype = torch.float32  # safe on M1

logger.info(f"Loading model {model_id} with dtype={torch_dtype}")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)
pipe.safety_checker = None  # optional

# ----- 4. Memory-efficient attention -----
pipe.enable_attention_slicing()

# ----- 5. Prompt and generation settings -----
prompt = "a futuristic NFL football field, ultra-detailed"
num_images = 3  # number of images to generate
num_steps = 25
guidance_scale = 7.5
height = 512
width = 512
seed = 42

logger.info(f"Generating {num_images} image(s) with {num_steps} steps and guidance_scale={guidance_scale}")

# ----- 6. Generate images one by one -----
start_time = time.time()

for i in range(num_images):
    generator = torch.Generator(device=device).manual_seed(seed + i)  # different seed for each image
    with torch.inference_mode():
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

    filename = f"output_{i}.png"
    image.save(filename)
    logger.success(f"Saved {filename}")

end_time = time.time()
elapsed = end_time - start_time
logger.success(f"All {num_images} images generated in {elapsed:.2f} seconds")
logger.info("Request Complete")
