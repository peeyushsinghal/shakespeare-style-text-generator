from model import GPT
from data import DataLoaderLite
import torch
from dataclasses import dataclass  # for dataclass, config class
from utils import get_device, set_seed


@dataclass
class Config:
    model_name: str = "gpt2"
    seed: int = 1337
    max_return_sequences: int = 5
    file_path: str = "data/input.txt"
    max_length: int = 30
    B: int = 8 # batch size
    T: int = 512 # sequence length
    lr: float = 1e-4 # learning rate
    epochs: int = 5000
    interval: int = 100
    moving_avg_window: int = 100
    best_loss: float = float('inf')
    checkpoint_dir: str = "checkpoint"
    target_loss: float = 0.099999


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    dropout: float = 0.1  # dropout rate
    bias: bool = True  # use bias in attention and feedforward



def load_model(model_type=None):
    if model_type is not None:
        model = GPT.from_pretrained(model_type=model_type)
    else:
        model_config = GPTConfig()
        model = GPT(model_config)
    return model

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("\nModel Architecture:")
    print("=" * 50)
    print(f"Block Size (Context Length): {model.config.block_size}")
    print(f"Vocabulary Size: {model.config.vocab_size}")
    print(f"Number of Layers: {model.config.n_layer}")
    print(f"Number of Heads: {model.config.n_head}")
    print(f"Embedding Dimension: {model.config.n_embd}")
    print(f"Dropout: {model.config.dropout}")
    
    # Calculate parameter counts for each component
    token_emb = model.config.vocab_size * model.config.n_embd
    pos_emb = model.config.block_size * model.config.n_embd
    
    # Per layer parameters
    attn_params = 4 * model.config.n_embd * model.config.n_embd  # Q,K,V, and output projection
    mlp_params = 8 * model.config.n_embd * model.config.n_embd   # MLP with 4x expansion
    layer_params = attn_params + mlp_params
    
    print("\nParameter Counts:")
    print("-" * 50)
    print(f"Token Embeddings: {token_emb:,}")
    print(f"Position Embeddings: {pos_emb:,}")
    print(f"Per Layer: {layer_params:,}")
    print(f"All Layers: {layer_params * model.config.n_layer:,}")
    print(f"Total Trainable Parameters: {count_parameters(model):,}")
    
    # Estimated model size
    model_size_mb = count_parameters(model) * 4 / (1024 * 1024)  # 4 bytes per parameter
    half_precision_size = model_size_mb / 2
    print(f"\nEstimated Model Size:")
    print(f"Full Precision (MB): {model_size_mb:.2f}")
    print(f"Half Precision (MB): {half_precision_size:.2f}")
    print("=" * 50 + "\n")


def main():

    config = Config()

    # set up device
    device = get_device()
    print(f"Using device: {device}")

    # set seed
    set_seed(config.seed)

    # load model
    # model = load_model(config.model_name) # from pretrained
    model = load_model()  # from scratch
    # Print model summary
    print_model_summary(model)
    model.to(device)

    # load dataset
    train_loader = DataLoaderLite(
        B=config.B, T=config.T, file_path=config.file_path, model_type=config.model_name
    )
    # print(train_loader.next_batch()) # check if data is loaded correctly

    # train model

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    losses = []
    best_loss = config.best_loss

    for i in range(config.epochs):
        x, y = train_loader.next_batch()
        optimizer.zero_grad()
        _, loss = model(x.to(device), y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to prevent exploding gradients
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        # Calculate moving average loss
        avg_loss = sum(losses[-config.moving_avg_window:]) / min(config.moving_avg_window, len(losses))

        if loss.item() < best_loss and i > config.interval-1:
            best_loss = loss.item()
            torch.save({"model_state_dict": model.state_dict(), 
                "config": config,
                "model_config":GPTConfig()}, 
                f"{config.checkpoint_dir}/model_best.pth")
            print(f"Model saved at step {i}")

        if i % config.interval == 0:
            print(f"step{i}, loss: {loss.item():.4f}, best loss: {best_loss:.4f}, moving avg loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < config.target_loss:
            print(f"---Target loss reached at step {i}")
            torch.save({"model_state_dict": model.state_dict(), 
                "config": config,
                "model_config":GPTConfig()}, 
                f"{config.checkpoint_dir}/model_final.pth")
            break
        
    print(f"Training completed. Best loss: {best_loss:.4f}, final loss: {loss.item():.4f}")
    # save model
    torch.save({"model_state_dict": model.state_dict(), 
                "config": config,
                "model_config":GPTConfig()}, 
                f"{config.checkpoint_dir}/model_final.pth")
    print(f"Model saved to {config.checkpoint_dir}/model_final.pth")

    # inference
    # print(model)
    return model


if __name__ == "__main__":
    model = main()
    print(model)