import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, logging as transformers_logging
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
import warnings
import logging
from Upsample import RealESRGAN
import sys

# Completely suppress all warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific loggers
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()

# Suppress PyTorch warnings
logging.getLogger("torch.serialization").setLevel(logging.ERROR)

# Print loading message
print("Loading Janus Pro 7B model... This may take a few minutes.")

# Load model and processor
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'

# Suppress stdout during model loading to hide tqdm progress bars
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True
    )
    vl_chat_processor = VLChatProcessor.from_pretrained(
        model_path,
        use_fast=True
    )
finally:
    sys.stdout.close()
    sys.stdout = original_stdout

print("Model loaded successfully!")

# Move model to appropriate device
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SR model
print("Loading super-resolution model...")
sr_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=True)
print("Super-resolution model loaded successfully!")

@torch.inference_mode()
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0,
                   num_images=5):  # Accept num_images as input
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = num_images  # Use num_images instead of hardcoded value
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        stime = time.time()
        ret_images = [image_upsample(Image.fromarray(images[i])) for i in range(parallel_size)]
        print(f'upsample time: {time.time() - stime}')
        return ret_images


def image_upsample(img: Image.Image) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    global sr_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if sr_model is None:
        sr_model = RealESRGAN(device, scale=2)
        sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=False)

    result = sr_model.predict(img.convert('RGB'))
    return result

# Custom CSS for styling
custom_css = """
:root {
    --main-bg-color: #0a0a0a;
    --card-bg-color: #141414;
    --header-color: #ffffff;
    --accent-color: #3a86ff;
    --secondary-color: #4361ee;
    --text-color: #e0e0e0;
    --border-radius: 12px;
    --shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    --input-bg: #1a1a1a;
    --input-border: #2a2a2a;
}

body {
    background-color: var(--main-bg-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    color: var(--header-color);
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, var(--accent-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header p {
    font-size: 1.1rem;
    color: var(--text-color);
    opacity: 0.8;
}

.tab-nav {
    background-color: var(--card-bg-color);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    overflow: hidden;
}

.tab-button {
    padding: 1rem 1.5rem;
    font-weight: 600;
    color: var(--text-color);
    border-bottom: 3px solid transparent;
    transition: all 0.3s ease;
}

.tab-button.selected {
    color: var(--accent-color);
    border-bottom-color: var(--accent-color);
}

.card {
    background-color: var(--card-bg-color);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--header-color);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.input-group {
    margin-bottom: 1rem;
}

.input-label {
    font-weight: 500;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

.button-primary {
    background: linear-gradient(135deg, var(--accent-color), var(--secondary-color));
    color: white;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border-radius: var(--border-radius);
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(58, 134, 255, 0.3);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.9rem;
}

.button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(58, 134, 255, 0.4);
}

.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}

.gallery-item {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: transform 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.gallery-item:hover {
    transform: scale(1.03);
}

.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #333;
    color: var(--text-color);
    opacity: 0.7;
    font-size: 0.9rem;
}

/* Override Gradio's default styles */
.gradio-container {
    background-color: var(--main-bg-color) !important;
    max-width: 100% !important;
}

.gradio-container input, 
.gradio-container textarea, 
.gradio-container select {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    color: var(--text-color) !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem !important;
}

.gradio-container input:focus, 
.gradio-container textarea:focus, 
.gradio-container select:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(58, 134, 255, 0.2) !important;
}

.gradio-container label {
    color: var(--text-color) !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}

.gradio-container .tabs {
    background-color: var(--card-bg-color) !important;
    border-radius: var(--border-radius) !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.gradio-container .tab-selected {
    color: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
    background-color: rgba(58, 134, 255, 0.1) !important;
}

.gradio-container .accordion {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: var(--border-radius) !important;
}

.gradio-container .accordion-header {
    padding: 0.75rem !important;
    font-weight: 500 !important;
}

.gradio-container .slider {
    background-color: rgba(255, 255, 255, 0.1) !important;
}

.gradio-container .slider-handle {
    background-color: var(--accent-color) !important;
}

.gradio-container .slider-track {
    background-color: var(--accent-color) !important;
}

.gradio-container .file-preview {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: var(--border-radius) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .container {
        padding: 1rem;
    }
    
    .header h1 {
        font-size: 2rem;
    }
}
"""

# Gradio interface with enhanced UI
with gr.Blocks(css=custom_css) as demo:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h1 style="font-size: 2.5rem; font-weight: 700; color: #ffffff;">Deepseek ai Janus Pro 7B</h1>
                    <p style="color: #e0e0e0; font-size: 1.2rem;">A powerful multimodal AI model for image understanding and generation</p>
                </div>
            """)
    
    # Tabs for different functionalities
    with gr.Tabs() as tabs:
        # Multimodal Understanding Tab
        with gr.TabItem("Image Understanding", elem_classes="tab-button"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="card"):
                    gr.HTML('<div class="card-title">Upload Image</div>')
                    image_input = gr.Image(
                        type="numpy",
                        label="",
                        height=400
                    )
                
                with gr.Column(scale=1, elem_classes="card"):
                    gr.HTML('<div class="card-title">Ask About the Image</div>')
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What can you tell me about this image?",
                        lines=3
                    )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            und_seed_input = gr.Number(
                                label="Seed",
                                precision=0,
                                value=42
                            )
                        
                        with gr.Row():
                            with gr.Column():
                                top_p = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=0.95,
                                    step=0.05,
                                    label="Top-p Sampling"
                                )
                            
                            with gr.Column():
                                temperature = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=0.1,
                                    step=0.05,
                                    label="Temperature"
                                )
                    
                    understanding_button = gr.Button(
                        "Analyze Image",
                        elem_classes="button-primary"
                    )
            
            with gr.Row(elem_classes="card"):
                gr.HTML('<div class="card-title">AI Response</div>')
                understanding_output = gr.Markdown(
                    label=""
                )
        
        # Text-to-Image Generation Tab
        with gr.TabItem("Image Generation", elem_classes="tab-button"):
            with gr.Column(elem_classes="card"):
                gr.HTML('<div class="card-title">Create Images from Text</div>')
                prompt_input = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Describe the image you want to generate in detail...",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column():
                        seed_input = gr.Number(
                            label="Seed (Optional)",
                            precision=0,
                            value=1234
                        )
                    
                    with gr.Column():
                        num_images_input = gr.Slider(
                            label="Number of Images",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1
                        )
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            cfg_weight_input = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=0.5,
                                label="CFG Weight (Guidance Scale)"
                            )
                        
                        with gr.Column():
                            t2i_temperature = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=1.0,
                                step=0.05,
                                label="Temperature"
                            )
                
                generation_button = gr.Button(
                    "Generate Images",
                    elem_classes="button-primary"
                )
            
            with gr.Column(elem_classes="card"):
                gr.HTML('<div class="card-title">Generated Images</div>')
                image_output = gr.Gallery(
                    label="",
                    columns=3,
                    rows=2,
                    height=500
                )
    
    # No footer to match the screenshot
    
    # Set up event handlers
    understanding_button.click(
        fn=multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature, num_images_input],
        outputs=image_output
    )

# Launch the app with a cleaner interface
print("Starting the Gradio interface...")
demo.launch(
)
print("Opening app in your browser...")
