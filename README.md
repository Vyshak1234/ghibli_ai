# ghibli_ai

🌟 Transform your images into Studio Ghibli-style artwork using Stable Diffusion! 🌟

🚀 Overview

This Python script utilizes the StableDiffusionImg2ImgPipeline from diffusers to convert input images into stunning Studio Ghibli-style artwork. It offers:

Hugging Face authentication for model access.

Image preprocessing (resizing and RGB conversion).

GPU support for fast inference.

Customizable transformation settings (strength, guidance scale, and variations).

Downloadable outputs.

🛠️ Installation

Ensure you have Python installed, then install the necessary dependencies:

pip install huggingface_hub diffusers transformers accelerate safetensors xformers torch matplotlib pillow google-colab

🔑 Authentication

You need a Hugging Face token to access the model:

Get your token from Hugging Face.

Enter your token when prompted.

Alternatively, set it as an environment variable:

export HF_TOKEN=your_token_here

📸 Usage

Run the script:

python ghibli.py

Workflow

Upload an image (JPG, PNG, etc.).

Enter a description for the transformation (e.g., "a castle in the sky").

Adjust parameters (optional):

Transformation strength (0.1 - 0.9)

Guidance scale (1 - 15)

Number of variations (1 - 4)

Seed for reproducibility

View and download the transformed images.

🖌️ Features

Ghibli-style prompt templates: Automatically enhances prompts for better artistic results.

CUDA support: Detects GPU and optimizes model settings.

User interaction: Simple CLI-based user input for customization.

Seed control: Enables reproducible outputs.

Auto-download: Saves and downloads generated images.

📌 Model Information

Model: nitrosocke/Ghibli-Diffusion

Hosted on Hugging Face Hub

Uses fp16 precision for efficiency on GPUs


🛑 Troubleshooting

Model not loading? Ensure you've accepted the model terms on Hugging Face.

CUDA out of memory? Try reducing num_images or use cpu mode.

Slow generation? Running on CPU is significantly slower than on GPU.

📜 License

This project is for educational and personal use. Follow Hugging Face’s terms for model usage.

🤝 Contributing

Feel free to enhance the script or report issues!

Happy transforming! 🎨✨
