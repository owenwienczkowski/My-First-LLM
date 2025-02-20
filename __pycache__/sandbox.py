import tiktoken
import torch
from utility import GPTModel


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "emb_drop_rate": 0.1, # Embedding dropout rate
    "short_drop_rate": 0.1, # Shortcut dropout rate
    "multi_drop_rate": 0.1, # Multi-Head attention dropout rate
    "qkv_bias": False      # Query-key-value bias
}


torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

from gpt_download import download_and_load_gpt2

# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
# model_name = "gpt2-small (124M)"  # Example model name
# NEW_CONFIG = GPT_CONFIG_124M.copy()
# NEW_CONFIG.update(model_configs[model_name])
# NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# gpt = GPTModel(NEW_CONFIG)
# gpt.eval();


from utility import load_weights_into_gpt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load_weights_into_gpt(gpt, params)
# gpt.to(device);
import os
import urllib.request
from utility import create_dataloader_v1


file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()


# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)
from utility import calc_loss_loader

# torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader
# train_loss = calc_loss_loader(train_loader, gpt, device)
# val_loss = calc_loss_loader(val_loader, gpt, device)

# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

# Exercise 5.6
# Readers are encouraged to experiment with GPT-2 models of different sizes,
# for example, the largest 1558M parameter model and compare the generated
# text to the 124M model we loaded in this chapter.

# settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
# model_name = "gpt2-xl (1558M)"

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_name = "gpt2-small (124M)"


NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

