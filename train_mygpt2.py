# train_mygpt2.py -- myGPT2 -- Ryan-W31

# Imports
# -----------------------------------
from dataclasses import dataclass
import math
import tiktoken
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
    self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

    # output projection
    self.c_proj = nn.Linear(config.n_embed, config.n_embed)
    self.c_proj.SCALING_COEFF = 1.0

    self.n_head = config.n_head
    self.n_embed = config.n_embed

    # 'triangular' mask for queries
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                            .view(1, 1, config.block_size, config.block_size)))
    
    # forward pass
  def forward(self, x):
    B, T, C = x.size() # get dimensions (batch size, sequence length, embedding dimensionality)

    # get key, query, and value projections for all heads in batch
    q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

    # change k, q, v dimensions; nh = 'number of heads' and hs = 'head size'
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # get attention weights; transpose to get (T, T) matrix in last two dimensions
    attn = (q @ k.transpose(-2, -1))  * (1.0 / math.sqrt(k.size(-1)))
    attn = attn.masked_fill(self.bias[:, : , :T, :T] == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)

    out = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) ===> (B, ng, T, hs)
    out = out.transpose(1, 2).contiguous().view(B, T, C) # concatenate head outputs

    out = self.c_proj(out) # output projection
    return out
# -----------------------------------------------------

# FeedForward Module
# -----------------------------------------------------
class FeedForward(nn.Module):

  # module initialization
  def __init__(self, config):
    super().__init__()

    self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed) # linear layer
    self.gelu = nn.GELU(approximate="tanh") # no reason to use tanh anymore, but GPT-2 used it
    self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed) # linear projection layer
    self.c_proj.SCALING_COEFF = 1.0


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
    x = x + self.attn(self.ln_1(x)) # x + self.attn() because of residual connection
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

    self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False) # final classifier (lm_head)

    # weight shared between embedding layer and pre-softmax linear layer
    self.transformer.wte.weight = self.lm_head.weight

    # initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    std = 0.02

    # initialize weights of linear and embedding layers
    if isinstance(module, (nn.Linear, nn.Embedding)):

      # scale the standard deviation of weights by the scaling coefficient
      if isinstance(module, nn.Linear) and hasattr(module, 'SCALING_COEFF'):
        std *= (2 * self.config.n_layer) ** -0.5

      torch.nn.init.normal_(module.weight, mean=0.0, std=std)

      if isinstance(module, nn.Linear) and module.bias is not None:
        torch.nn.init.zeros_(module.bias)

  # forward pass
  def forward(self, idx, targets=None):
    B, T = idx.size() # idx is shape (B, T)

    # confirm that sequence length is less than or equal to block_size
    assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block_size is {self.config.block_size}'

    # get positional and token embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # positional embeddings of shape (T, n_embed)
    tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
    x = tok_emb + pos_emb

    # forward through all transformer blocks
    for block in self.transformer.h:
      x = block(x)

    # forward through last layer normalization and linear layer
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)

    loss = None
    if targets is not None:
      # flattening out (B, T, vocab_size) to (B, T) and (B, T) to (B*T)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss

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
          sd[k].copy_(sd_hf[k].t())

      else:
        # copy over other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    
    return model
# -----------------------------------------------------

# Data Loader
# -----------------------------------------------------
class DataLoader:

  # module initialization
  def __init__(self, B, T):
    self.B = B # batch size
    self.T = T # sequence length

    # initialize tokens and store them in memory
    with open('/kaggle/input/mygpt2-train/input.txt', 'r') as f:
      text = f.read()
    # with open('input.txt', 'r') as f:
    #   text = f.read()

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text) # list of tokens
    self.tokens = torch.tensor(tokens) # (N,)
    self.curr_pos = 0 # current position in the tokens

    print(f'Number of tokens: {len(self.tokens)}')
    print(f'Number of batches: {len(self.tokens) // (B * T)}')

  # get next batch
  def next_batch(self):
    B, T = self.B, self.T # batch size, sequence length
    buffer = self.tokens[self.curr_pos : self.curr_pos + B * T + 1] # (B * T + 1,)
    x = buffer[:-1].view(B, T) # (B, T) inputs
    y = buffer[1:].view(B, T) # (B, T) targets

    self.curr_pos += B * T # move the current position

    # if we reach the end of the tokens, reset the position
    if self.curr_pos + B * T + 1 >= len(self.tokens):
      self.curr_pos = 0
    return x, y
    

# Change device as needed
# -----------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  device = "mps"
print(f'Using device: {device}')
# -----------------------------------------------------

torch.manual_seed(42) # for reproducibility
if torch.cuda.is_available():
  torch.cuda.manual_seed(42) # for reproducibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  torch.mps.manual_seed(42) # for reproducibility

train_loader = DataLoader(B=4, T=32)
num_return_sequences = 5
max_length = 30

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
print("Woo Hoo! Didn't crash!")

model.eval()
model.to(device)

# Optimization
# -----------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for i in range(50):
  x, y = train_loader.next_batch()
  x, y = x.to(device), y.to(device)
  optimizer.zero_grad()
  logits, loss = model(x, y)
  loss.backward()
  optimizer.step()
  print(f'step: {i:2d}, loss: {loss.item():.4f}')

import sys; sys.exit(0)
# -----------------------------------------------------

# Model Generation
# -----------------------------------------------------

if device == 'cuda':
  torch.cuda.manual_seed(42) # for reproducibility
elif device == 'mps':
  torch.mps.manual_seed(42) # for reproducibility

torch.manual_seed(42) # for reproducibility

while x.size(1) < max_length: # ensure size does not excede max_length

  # forward the model
  with torch.no_grad(): # don't keep track of gradients in these operations

    logits = model(x) # get logits from model (B, T, vocab_size)

    logits = logits[:, -1, :] # get logits at the last position (B, vocab_size)

    probs = F.softmax(logits, dim=-1) # get probabilities (B, vocab_size)

    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # top-k sampling of 50 (B, 50)

    ix = torch.multinomial(topk_probs, num_samples=1) # get token from topk_probs (B, 1)
    xcol = torch.gather(topk_indices, -1, ix) # get corresponding indices (B, 1)

    x = torch.cat((x, xcol), dim=1) # append to the sequence

# print generated text
for i in range(num_return_sequences):
  tokens = x[i, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(">", decoded)