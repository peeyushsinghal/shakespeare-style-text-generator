import gradio as gr
from infer import load_model_from_checkpoint, generate_text, InferenceConfig
from utils import get_device
from main import GPTConfig, Config
from torch.serialization import add_safe_globals
from dataclasses import dataclass



import warnings
# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)



@dataclass
class AppConfig:
    model_path: str = "checkpoint/model_final.pth"
    num_return_sequences: int = 5
    max_length: int = 50  # max length of the generated text
    tokenizer: str = "gpt2"

config = AppConfig()

device = get_device()
add_safe_globals([Config, GPTConfig])

model = load_model_from_checkpoint(config.model_path, device=device)

def generate(prompt, num_sequences):
    if not prompt:
        return "Please enter a prompt."
    
    generated_texts = generate_text(
        model=model,
        prompt=prompt,
        num_return_sequences=num_sequences,
        device=device
    )
    
    # Format output with sequence numbers
    formatted_output = ""
    for i, text in enumerate(generated_texts, 1):
        formatted_output += f"**Sequence {i}**:\n{text}\n\n"
    
    return formatted_output

# Create Gradio interface
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            lines=3, 
            placeholder="Enter your prompt here...",
            label="Prompt"
        ),
        gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=3,
            label="Number of Sequences"
        )
    ],
    outputs=gr.Textbox(
        lines=10,
        label="Generated Text"
    ),
    title="Text Generation with GPT",
    description="Enter a prompt and select the number of sequences to generate different variations of text.",
)

if __name__ == "__main__":
    iface.launch() 