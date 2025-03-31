!pip install huggingface_hub
!pip install diffusers transformers accelerate safetensors
!pip install xformers


import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io
import warnings
from huggingface_hub import login
import os
import time
warnings.filterwarnings("ignore")

# First, authenticate with Hugging Face
print("=" * 70)
print("🌟 Ghibli Style Image Transformer 🌟")
print("=" * 70)
print("\nYou need to authenticate with Hugging Face to access the Ghibli-Diffusion model")
print("Go to https://huggingface.co/settings/tokens to create a token if you don't have one")

# Allow for token to be stored in env variable for convenience
hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token:
    hf_token = input("Enter your Hugging Face token: ")
login(token=hf_token)

# Function to upload an image
def upload_image():
    print("\n📤 Please upload an image...")
    uploaded = files.upload()

    for filename in uploaded.keys():
        img_bytes = uploaded[filename]
        img = Image.open(io.BytesIO(img_bytes))
        return img, filename

    return None, None

# Function to resize image to appropriate dimensions
def resize_image(img, max_size=768):
    width, height = img.size

    # Calculate new dimensions while maintaining aspect ratio
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = width
            new_height = height
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = width
            new_height = height

    # Resize only if needed
    if width > max_size or height > max_size:
        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        img = img.resize((new_width, new_height), Image.LANCZOS)

    return img

# Function to convert image to RGB if needed
def ensure_rgb(img):
    if img.mode != "RGB":
        print(f"Converting image from {img.mode} to RGB")
        return img.convert("RGB")
    return img

# Function to check for CUDA availability and print memory info
def check_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_free = mem_total - mem_reserved

        print(f"✅ Using GPU: {device_name}")
        print(f"   Total memory: {mem_total:.2f} GB")
        print(f"   Free memory: {mem_free:.2f} GB")
        return "cuda", True
    else:
        print("⚠️ WARNING: Running on CPU. This will be very slow!")
        return "cpu", False

# Try loading the model with authentication
def load_model(hf_token):
    device, has_cuda = check_gpu()
    print("\n⏳ Loading Ghibli-Diffusion model...")
    start_time = time.time()

    try:
        model_id = "nitrosocke/Ghibli-Diffusion"

        # Load with appropriate settings based on device
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            use_safetensors=True,
            variant="fp16" if has_cuda else None,
            token=hf_token
        )

        # Enable memory efficient attention if CUDA is available
        if has_cuda:
            pipe.enable_xformers_memory_efficient_attention()
            # Alternatively, if xformers isn't available:
            # pipe.enable_attention_slicing(slice_size="auto")

        pipe = pipe.to(device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
        return pipe, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease check:")
        print("1. You're logged in to Hugging Face and have accepted the model terms")
        print("2. You're using a valid token with proper permissions")
        print("3. You have sufficient GPU memory (try restarting your runtime)")
        return None, device

# Enhanced Ghibli style prompts templates
ghibli_style_prompts = [
    "Studio Ghibli style art of {}, watercolor painting, hand-drawn animation, Hayao Miyazaki, soft lighting, colorful landscape",
    "Anime artwork in the style of Studio Ghibli, {} with dreamy atmosphere, painterly style, whimsical, detailed background",
    "{} in Studio Ghibli aesthetic, magical realism, hand-painted textures, gentle colors, fantasy world, artistic animation still",
    "Ghibli-inspired illustration of {}, pastel colors, Totoro style, whimsical fantasy, detailed environment",
    "{} rendered in Miyazaki's signature style, spiritual elements, nature-focused, hand-painted animation cel"
]

def generate_ghibli_image(pipe, device, input_image, prompt_addition="", strength=0.75, guidance_scale=7.5, num_images=1, seed=None):
    """
    Generate Ghibli-style image from input image

    Args:
        pipe: The StableDiffusionImg2ImgPipeline
        device: Device to use (cuda/cpu)
        input_image: PIL Image to transform
        prompt_addition: Additional text to add to the base Ghibli prompt
        strength: How much to transform the image (0-1)
        guidance_scale: How closely to follow the prompt
        num_images: Number of images to generate
        seed: Random seed for reproducibility

    Returns:
        List of generated images and the prompt used
    """
    # Prepare the image
    input_image = ensure_rgb(input_image)
    input_image = resize_image(input_image)

    # Choose a random Ghibli style prompt template and insert the user's prompt
    import random
    template = random.choice(ghibli_style_prompts)
    base_prompt = template.format(prompt_addition)

    print(f"\n🖌️ Transforming image with prompt: \"{base_prompt}\"")
    print(f"   Strength: {strength} | Guidance Scale: {guidance_scale}")

    # Set the generator for reproducibility if seed is provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"   Using seed: {seed}")

    # Generate images
    start_time = time.time()
    with torch.no_grad():
        output = pipe(
            prompt=base_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
            negative_prompt="low quality, blurry, distorted, deformed, disfigured, bad anatomy, watermark"
        )

    generation_time = time.time() - start_time
    print(f"✅ Image generated in {generation_time:.2f} seconds")

    return output.images, base_prompt

# Function to display images side by side
def display_images(original, generated, titles=None):
    n_generated = len(generated)
    fig_width = 5 * (n_generated + 1)
    fig_height = 5

    plt.figure(figsize=(fig_width, fig_height))

    # Display original
    plt.subplot(1, n_generated + 1, 1)
    plt.imshow(np.array(original))
    plt.title(titles[0] if titles else "Original Image")
    plt.axis("off")

    # Display generated
    for i, img in enumerate(generated):
        plt.subplot(1, n_generated + 1, i + 2)
        plt.imshow(np.array(img))
        plt.title(titles[i+1] if titles and i+1 < len(titles) else f"Ghibli Style {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Save and download images
def save_images(generated_images, base_filename):
    saved_files = []
    for i, img in enumerate(generated_images):
        output_filename = f"ghibli_{base_filename.split('.')[0]}_{i}.png"
        img.save(output_filename)
        files.download(output_filename)
        saved_files.append(output_filename)
        print(f"📥 Image {i+1} saved and downloaded as {output_filename}")
    return saved_files

# Main function
def main():
    print("\n🌸 Welcome to Ghibli Style Image Generator! 🌸")
    print("=" * 70)
    print("This program will transform your images into Studio Ghibli style artwork.")

    # Load model
    pipe, device = load_model(hf_token)
    if pipe is None:
        print("Failed to load model. Please restart and try again.")
        return

    while True:
        # Upload image
        input_image, filename = upload_image()
        if input_image is None:
            print("No image uploaded. Please try again.")
            continue

        # Display original image
        plt.figure(figsize=(5, 5))
        plt.imshow(np.array(input_image))
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

        # Get additional prompt from user
        prompt_addition = input("\n✨ Enter description for the Ghibli style (what's in the image): ")

        # Get strength
        try:
            strength_input = input("🔄 Enter transformation strength (0.1-0.9, default: 0.75): ")
            strength = float(strength_input) if strength_input.strip() else 0.75
            strength = max(0.1, min(0.9, strength))  # Clamp between 0.1 and 0.9
        except:
            strength = 0.75
            print("Using default strength: 0.75")

        # Get guidance scale
        try:
            guidance_input = input("🎯 Enter guidance scale (1-15, default: 7.5): ")
            guidance_scale = float(guidance_input) if guidance_input.strip() else 7.5
            guidance_scale = max(1.0, min(15.0, guidance_scale))
        except:
            guidance_scale = 7.5
            print("Using default guidance scale: 7.5")

        # Get number of images
        try:
            num_input = input("🖼️ How many variations to generate? (1-4, default: 1): ")
            num_images = int(num_input) if num_input.strip() else 1
            num_images = max(1, min(4, num_images))
        except:
            num_images = 1
            print("Generating 1 image")

        # Get seed for reproducibility
        try:
            seed_input = input("🎲 Enter seed for reproducibility (leave blank for random): ")
            seed = int(seed_input) if seed_input.strip() else None
        except:
            seed = None

        # Generate images
        print("\n⏳ Generating Ghibli style image(s)... please wait...")
        generated_images, used_prompt = generate_ghibli_image(
            pipe,
            device,
            input_image,
            prompt_addition=prompt_addition,
            strength=strength,
            guidance_scale=guidance_scale,
            num_images=num_images,
            seed=seed
        )

        # Display results
        display_images(
            input_image,
            generated_images,
            titles=["Original Image"] + [f"Ghibli Style {i+1}" for i in range(len(generated_images))]
        )

        # Save images
        saved_files = save_images(generated_images, filename)

        print(f"\n📝 Prompt used: \"{used_prompt}\"")
        if seed is not None:
            print(f"🎲 Seed used: {seed}")

        # Ask if user wants to generate another image
        another = input("\n🔄 Generate another image? (y/n): ").lower().strip()
        if another != 'y':
            print("\n👋 Thank you for using the Ghibli Style Image Generator! Goodbye!")
            break

if __name__ == "__main__":
    main()
