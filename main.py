import torch
from gpt import GPTModel, generate_text_simple
from tokenizer import create_tokenizer, create_dataloader
import tiktoken

USE_TIKTOKEN = False

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 1024,      # Context length
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-Key-Value bias
}
GPT_CONFIG_CUSTOM = {
    "vocab_size": 0,      # Vocabulary size, fill in later
    "ctx_len": 1024,      # Context length
    "emb_dim": 768,       # Embedding dimension
    "n_heads": 12,        # Number of attention heads
    "n_layers": 12,       # Number of layers
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-Key-Value bias
}

tokenizer = {}
gpt_model = {}
if USE_TIKTOKEN:
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt_model = GPTModel(GPT_CONFIG_124M)
else:
    tokenizer = create_tokenizer()
    GPT_CONFIG_CUSTOM["vocab_size"] = tokenizer.vocab_size()
    gpt_model = GPTModel(GPT_CONFIG_CUSTOM)





# disable dropout
gpt_model.eval()

encoded_text = tokenizer.encode("The quick brown fox jumped over the lazy dog")
encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)

out = generate_text_simple(model=gpt_model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["ctx_len"])
print("out: ", out)

out_text = tokenizer.decode(out.squeeze(0).tolist())
print("out_text: ", out_text)

