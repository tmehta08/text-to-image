import torch
from diffusers import StableDiffusionPipeline
from loguru import logger
import time
import sys

# ----- 1. Configure logger -----
logger.remove()  # Remove default logger
logger.add(sys.stdout, level="INFO")  # console logging
logger.add("generation.log", rotation="10 MB")  # log to file
logger.info("Starting Stable Diffusion pipeline")

# ----- 2. Device -----
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logger.info(f"Using device: {device}")

# ----- 3. Load model -----
model_id = "runwayml/stable-diffusion-v1-5"
torch_dtype = torch.float16 if device.type == "mps" else torch.float32

logger.info(f"Loading model {model_id} with dtype={torch_dtype}")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# Optional: Disable NSFW safety checker to speed up generation
pipe.safety_checker = None

# ----- 4. Memory-efficient attention -----
pipe.enable_attention_slicing()  # slices attention for lower VRAM
# pipe.enable_xformers_memory_efficient_attention()  # optional if xformers installed

# ----- 5. Prompt and generation settings -----
prompt = "a futuristic NFL football field, ultra-detailed, 4k"
batch_size = 2
num_steps = 25
guidance_scale = 7.5
generator = torch.Generator(device=device).manual_seed(42)

logger.info(f"Generating {batch_size} images with {num_steps} steps and guidance_scale={guidance_scale}")

# ----- 6. Timing the generation -----
start_time = time.time()

# Use inference mode and autocast for MPS acceleration
with torch.inference_mode():
    with torch.autocast(device.type):
        images = pipe(
            [prompt] * batch_size,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images

end_time = time.time()
elapsed = end_time - start_time
logger.success(f"Image generation completed in {elapsed:.2f} seconds")

# ----- 7. Save outputs -----
for i, img in enumerate(images):
    filename = f"output_{i}.png"
    img.save(filename)
    logger.success(f"Saved {filename}")

logger.info("Request Complete")
