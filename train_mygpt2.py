# train_mygpt2.py -- myGPT2 -- Ryan-W31

# Imports
# -----------------------------------
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------

# Hyperparameters
# -----------------------------------------------------
batch_size = 64 # how many independent sequences to process in each batch
block_size = 256 # length of each sequence in each batch
max_iters = 5000 # num of iterations in training loop
eval_interval = 500 # evaluate model every 500 iterations
learning_rate = 3e-4 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use CUDA if available
eval_iters = 200 # num iterations in evaluation loop
n_embed = 384 # num of embeddings
n_head = 6 # num of self-attention heads
n_layer = 6 # num of layers
dropout = 0.2 # dropout rate
# -----------------------------------------------------

# CausalSelfAttention Module
# -----------------------------------------------------
class CausalSelfAttention(nn.Module):

  # module initialization
  def __init__(self, config):
    super().__init__()
    assert config.n_embed % config.n_head == 0 # ensure head size will be an integer

    # key, query, and value projections (3 * n_embed)
    self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)

    # output projection
    self.c_proj = nn.Linear(config.n_embed, config.n_embed)

    self.n_head = config.n_head
    self.n_embed = config.n_embed

    # 'triangular' mask for queries
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                            .view(1, 1, config.block_size, config.block_size)))
    
    # forward pass
    def forward(x):
      B, T, C = x.size() # get dimensions (batch size, sequence length, embedding dimensionality)

      # get key, query, and value projections for all heads in batch
      k, q, v = self.c_attn(x).split(self.n_embed, dim=2)

      # change k, q, v dimensions; nh = 'number of heads' and hs = 'head size'
      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

      # get attention weights; transpose to get (T, T) matrix in last two dimensions
      attn = (q @ k.transpose(-2, -1))  * (1 / math.sqrt(k.size(-1)))
      attn = attn.masked_fill(self.bias[:, : , :T, :T] == 0, float("-inf"))
      attn = F.softmax(attn, dim=-1)

      out = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) ===> (B, ng, T, hs)
      out.transpose(1, 2).contiguous().view(B, T, C) # concatenate head outputs

      out = self.c_proj(out) # output projection
      return out
# -----------------------------------------------------

# FeedForward Module
# -----------------------------------------------------
class FeedForward(nn.Module):

  # module initialization
  def __init__(self, config):
    super().__init__()

    self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed), # linear layer
    self.gelu = nn.GELU(approximate="tanh"), # no reason to use tanh anymore, but GPT-2 used it
    self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed), # linear projection layer

  # forward pass
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x
# -----------------------------------------------------

# Block Module
# -----------------------------------------------------
class Block(nn.Module):

  # module initialization
  def __init__(self, config):
    super().__init__()
    self.attn = CausalSelfAttention(config) # multi-head self attention module (sa)
    self.mlp = FeedForward(config) # feed forward module (mlp)
    self.ln_1 = nn.LayerNorm(config.n_embed) # layer normalization before self-attention (ln1)
    self.ln_2 = nn.LayerNorm(config.n_embed) # layer normalization before feed forward (ln2)

  # forward pass
  def forward(self, x):
    x = x + self.attn(self.ln_1(x)) # x + self.sa() because of residual connection
    x = x + self.mlp(self.ln_2(x)) # x + self.mlp() because of residual connection

    return x
# -----------------------------------------------------

# GPT Configuration
# -----------------------------------------------------
@dataclass
class GPTConfig:
  block_size: int = 1024 # max sequence length
  vocab_size: int = 50257 # number of tokens
  n_layer: int = 12 # number of layers
  n_head: int = 12 # number of heads
  n_embed: int = 768 # embedding dimension
# -----------------------------------------------------


# Main GPT Module
# -----------------------------------------------------
class GPT(nn.Module):

  # module initialization
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embed), # weights of token embedding (wte)
      wpe = nn.Embedding(config.block_size, config.n_embed), # weights of positional embedding (wpe)
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers (h)
      ln_f = nn.LayerNorm(config.n_embed) # final layer-normalization (ln_f)
    ))
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # final classifier (lm_head)

  @classmethod
  def from_pretrained(cls, model_type):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("Loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head, n_embed are determined by model type
    config_args = {
      'gpt2':         dict(n_layer = 12, n_head = 12, n_embed = 768),  # 124M parameters
      'gpt2-medium':  dict(n_layer = 24, n_head = 16, n_embed = 1024), # 350M parameters
      'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embed = 1280), # 447M parameters
      'gpt2-xl':      dict(n_layer = 48, n_head = 25, n_embed = 1600), # 1558M parameters
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 tokens
    config_args['block_size'] = 1024  # always 1024 sequence length

    # creating from scratch initialized GPT-2 model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

    # initialize a huggingface/transformer model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring parameters are aligned and match in name and shape
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [ k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

    # openai checkpoints use 'Conv1d' module, this means we must transpose these weights
    assert len(sd_keys_hf) == len(sd_keys), f'Mismatched Keys {len(sd_keys_hf)} != {len(sd_keys)}'
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # transpose Conv1d weights
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy(sd_hf[k].t())

      else:
        # copy over other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy(sd_hf[k])
    
    return model
# -----------------------------------------------------

model = GPT.from_pretrained('gpt2')
print("Woo Hoo! Didn't crash!")