# Janus Pro 7B - Multimodal AI Model

Janus Pro 7B is a powerful multimodal AI model designed for advanced image understanding and text-to-image generation. This project leverages the `deepseek-ai/Janus-Pro-7B` model, enabling high-quality visual and textual interactions.
![Screenshot 2025-03-10 232506](https://github.com/user-attachments/assets/0f2caa6a-4232-4587-84b4-85c40dbb600e)

## Features
- **Multimodal Understanding:** Analyze images and answer questions based on visual content.
- **Text-to-Image Generation:** Generate images from textual descriptions using AI.
- **Super-Resolution Support:** Enhance image resolution using RealESRGAN.
- **Customizable Parameters:** Adjust settings such as seed, temperature, and top-p sampling for better results.
- **Optimized Performance:** Runs on GPU with reduced memory footprint and suppressed warnings for a smooth experience.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch with CUDA support
- Gradio for the interactive UI
- Transformers and other required libraries

### Install Dependencies
```sh
pip install torch torchvision torchaudio transformers gradio numpy pillow
pip install -r requirements.txt
```

## Usage
### 1. Run the Application
Execute the following command to launch the Gradio-based UI:
```sh
python app.py
```

### 2. Image Understanding
- Upload an image.
- Ask a question related to the image.
- Get AI-generated responses based on the visual content.

### 3. Text-to-Image Generation
- Provide a textual description.
- Adjust parameters such as seed and guidance weight.
- Generate high-quality images using the AI model.

### 4. Super-Resolution
- Enhance images using RealESRGAN for improved clarity and resolution.

## Model Details
- **Model:** `deepseek-ai/Janus-Pro-7B`
- **Processing:** Uses VLChatProcessor for input preparation.
- **Generation:** Supports controlled sampling and token generation for better results.

## Performance Optimization
- **CUDA Support:** Moves computation to GPU if available.
- **Memory Management:** Clears CUDA cache before processing to avoid memory overhead.
- **Suppressed Warnings:** Reduces log noise for a cleaner output.

## Customization
Modify parameters in `app.py` to fine-tune the model’s behavior:
- `temperature`: Controls randomness in generation.
- `top_p`: Adjusts nucleus sampling.
- `seed`: Ensures reproducibility of results.

## Acknowledgments
This project utilizes:
- [DeepSeek AI](https://deepseek.ai)
- [Transformers](https://huggingface.co/docs/transformers/)
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)

## License
This project is open-source and licensed under the MIT License.

---
For any questions or contributions, feel free to open an issue or submit a pull request.

