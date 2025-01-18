import tiktoken
from dataclasses import dataclass
import torch
from utils import get_device, set_seed
from main import GPTConfig, Config
from torch.serialization import add_safe_globals

from model import GPT

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class InferenceConfig:
    model_path: str = "../checkpoint/model_final.pth"
    num_return_sequences: int = 5
    max_length: int = 100  # max length of the generated text
    tokenizer: str = "gpt2"


config = InferenceConfig()


def encode(text, device, config=config):
    enc = tiktoken.get_encoding(config.tokenizer)
    enc_tensor = torch.tensor(enc.encode(text), dtype=torch.long, device=device)
    enc_tensor = enc_tensor.unsqueeze(0)
    return enc_tensor


def decode(tokens): ...


def generate_text(
    model,
    prompt,
    max_length=config.max_length,
    num_return_sequences=config.num_return_sequences,
    device=get_device(),
): 
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0).repeat(num_return_sequences, 1)
    input_ids = input_ids.to(device)

    #Calculate final length
    final_length = input_ids.shape[1] + max_length

    #Generate text
    with torch.no_grad():
        while input_ids.shape[1] < final_length:
            logits = model(input_ids)[0]
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)  
    

    generated_text = []
    for i in range(num_return_sequences):
        tokens = input_ids[i].tolist()
        text = tokenizer.decode(tokens)
        generated_text.append(text)

    return generated_text


def load_model_from_checkpoint(model_path, device):
    # Add Config and GPTConfig to safe globals
    add_safe_globals([Config, GPTConfig])

    try:
        # First try with weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        # If that fails, try without weights_only
        checkpoint = torch.load(model_path, map_location=device)

    # Get the model configuration from the saved GPTConfig
    model_config = checkpoint["model_config"]

    # Create a new model with this configuration
    model = GPT(model_config)

    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    return model


def inference():
    device = get_device()

    try:
        model = load_model_from_checkpoint(config.model_path, device=device)
        print("Model loaded successfully")
        # print(model)
        # return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    context = "To be or not to be, that is the question. "
    generated_text = generate_text(model, context)
    for text in generated_text:
        print(text)


if __name__ == "__main__":
    inference()
