# train_mygpt2.py -- myGPT2 -- Ryan-W31

# simple launch command:
# python train_mygpt2.py
# -----------------------------------
# DDP launch command:
# torchrun --standalone --nproc_per_node=<number-of-gpus> train_mygpt2.py

# Imports
# -----------------------------------
from dataclasses import dataclass
import inspect
import math
import numpy as np
import os
import tiktoken
import time
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from eval import render_example, iterate_examples
# -----------------------------------

# Hyperparameters
# -----------------------------------------------------
max_lr = 6e-4 # following GPT-3 configuration
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
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
    # attn = (q @ k.transpose(-2, -1))  * (1.0 / math.sqrt(k.size(-1)))
    # attn = attn.masked_fill(self.bias[:, : , :T, :T] == 0, float("-inf"))
    # attn = F.softmax(attn, dim=-1)
    # out = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) ===> (B, ng, T, hs)

    # switch to flash attention
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
  
  # optimizer configuration
  def configure_optim(self, weight_decay, lr, device_type):
    # get all candidate parameters (the ones that require grad)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # split up params so that all 2D and up tensors will be decayed, but 1D and constants will not
    # so weight tensors in matmul + embeddings decay, biases, and layernorms dont get decayed
    decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
    nondecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {'params' : decay_params, 'weight_decay': weight_decay},
      {'params' : nondecay_params, 'weight_decay': 0.0}
    ]

    # get number of decayed and non-decayed tensor and the number of parameters
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nondecay_params = sum(p.numel() for p in nondecay_params)
    print(f'# of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters.')
    print(f'# of non-decayed parameter tensors: {len(nondecay_params)}, with {num_nondecay_params:,} parameters.')

    # finally create an AdamW optimizer with the new parameter group and fused option if it's available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'

    # GPT-3 configuration
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) 
    return optimizer
# -----------------------------------------------------

# Data Loader
# -----------------------------------------------------

def load_tokens(filename):
  npt = np.load(filename) # get tokens from .npy file
  npt = npt.astype(np.int32)
  ptt = torch.tensor(npt, dtype=torch.long) # convert numpy array to torch tensor
  return ptt

class DataLoader:

  # module initialization
  def __init__(self, B, T, process_rank, num_processes, split):
    self.B = B # batch size
    self.T = T # sequence length
    self.process_rank = process_rank # which GPU is being used
    self.num_processes = num_processes # total number of GPUs being used

    assert split in {'train', 'val'}

    # initialize shards
    data_root = "fineweb_edu_10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards

    assert len(shards) > 0, f"No shards found for split {split}"

    if master_process:
      print(f"Found {len(shards)} shards for split {split}")
    
    self.reset()

  def reset(self):
    self.curr_shard = 0 # initialize at shard 0
    self.tokens = load_tokens(self.shards[self.curr_shard]) # get tokens from current shard
    self.curr_pos = self.B * self.T * self.process_rank # current position in the tokens

  # get next batch
  def next_batch(self):
    B, T = self.B, self.T # batch size, sequence length
    buffer = self.tokens[self.curr_pos : self.curr_pos + B * T + 1] # (B * T + 1,)
    x = buffer[:-1].view(B, T) # (B, T) inputs
    y = buffer[1:].view(B, T) # (B, T) targets

    self.curr_pos += B * T * self.num_processes # move the current position

    # if we reach the end of the tokens, load the next shard and reset the position
    if self.curr_pos + (B * T * self.num_processes + 1) >= len(self.tokens):
      self.curr_shard = (self.curr_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.curr_shard])
      self.curr_pos = B * T * self.process_rank

    return x, y
# -----------------------------------------------------

# Distributed Data Parallel (DDP) - using multiple GPUs in parallel
# -----------------------------------------------------

# torchrun is used to set env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # determine if this is a ddp capable run

if ddp:
  # DDP requires CUDA, devices are set based on rank
  assert torch.cuda.is_available(), "Must be using CUDA for DDP"
  init_process_group(backend="nccl")
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{ddp_local_rank}' # use appropriate GPU based on rank
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # logging and checkpointing will happen on this GPU
else:
  # no DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True

  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
  print(f'Using device: {device}')

device_type = "cuda" if device.startswith("cuda") else "cpu"
# -----------------------------------------------------

# Initialize model
# -----------------------------------------------------
torch.manual_seed(42) # for reproducibility
if torch.cuda.is_available():
  torch.cuda.manual_seed(42) # for reproducibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  torch.mps.manual_seed(42) # for reproducibility

enc = tiktoken.get_encoding('gpt2')

total_batch_size = 524288 # 2**19, GPT-3 paper used 0.5M
B = 16 # micro-batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "Ensure total_batch_size is divisible by B * T * ddp_world_size"
gradient_accum_step = total_batch_size // (B * T * ddp_world_size) # (2**19) // (16 * 1024) = 32
if master_process:
  print(f'Total desired batch size: {total_batch_size}')
  print(f'===> Calculated gradient accumulation steps: {gradient_accum_step}')

train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")


torch.set_float32_matmul_precision("high") # for efficiency

model = GPT(GPTConfig(vocab_size=50304)) # override vocab_size for better number
model.to(device)

use_compile = False # torch.compile breaks eval and sampling right now
if use_compile:
  model = torch.compile(model)

if ddp:
  model = DDP(model, device_ids={ddp_local_rank})

raw_model = model.module if ddp else model
#print("Woo Hoo! Didn't crash!")
# -----------------------------------------------------


# Learning Rate Scheduler
# -----------------------------------------------------


def get_lr(it):
  if it < warmup_steps: # linear warmup for warmup iterations
    return max_lr * (it + 1) / warmup_steps
  
  if it > max_steps: # if current iteration is larger than max_steps, return min learning rate
    return min_lr
  
  # in between warmup_steps and max_steps, use cosine decay to set learning rate all the way to min_lr
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1

  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr) 
# -----------------------------------------------------

# Hellaswag helper function
def get_most_likely_row(tokens, mask, logits):
    # evaluate the loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # now get the average loss where mask == 1 in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask

    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    pred_norm = avg_loss.argmin().item()  # the one with the lowest loss should be the most likely
    return pred_norm


# Optimization
# -----------------------------------------------------
optimizer = raw_model.configure_optim(weight_decay=0.1, lr=max_lr, device_type=device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, "w") as f: # clear file by opening in "write" mode
  pass

for step in range(max_steps):
  t0 = time.time() # step start time
  last_step = step == max_steps - 1 # get last step

  # model evaluation every 100 steps
  if step % 100 == 0:
    model.eval() # model to evaluation mode
    val_loader.reset() # reset data loader

    with torch.no_grad(): # don't keep track of gradients
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # type casting
          logits, loss = model(x, y) # forward pass
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    if ddp:
      dist.allreduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f"Validation Loss: {val_loss_accum.item():.4f}")

      # checkpointing
      with open(log_file, "a") as f:
        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
      if step > 0 and (step % 5000 == 0 or last_step):
          checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
          checkpoint = {
              'model': raw_model.state_dict(),
              'config': raw_model.config,
              'step': step,
              'val_loss': val_loss_accum.item()
          }
          torch.save(checkpoint, checkpoint_path)
  
  # occasionally evaluate using hellaswag
  if (step % 250 == 0 or last_step) and (not use_compile):
      num_correct_norm = 0
      num_total = 0
      for i, example in enumerate(iterate_examples("val")):
          if i % ddp_world_size != ddp_rank: # only process examples for this GPU
              continue
          
          _, tokens, mask, label = render_example(example) # get tokens, mask, and labels
          tokens = tokens.to(device)
          mask = mask.to(device)

          with torch.no_grad():
              with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                  logits, loss = model(tokens) # forward pass

              pred_norm = get_most_likely_row(tokens, mask, logits)

          num_total += 1
          num_correct_norm += int(pred_norm == label)

      if ddp:
          num_total = torch.tensor(num_total, dtype=torch.long, device=device)
          num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
          dist.all_reduce(num_total, op=dist.ReduceOp.SUM) # reduce the stats across all processes
          dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM) # reduce the stats across all processes
          num_total = num_total.item()
          num_correct_norm = num_correct_norm.item()
      acc_norm = num_correct_norm / num_total
      if master_process:
          print(f"HellaSwag Accuracy: {num_correct_norm} / {num_total} = {acc_norm:.4f}")
          with open(log_file, "a") as f:
              f.write(f"{step} HELLASWAG {acc_norm:.4f}\n")


  # occasionally sample from model
  if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while x.size(1) < max_length: # ensure size does not excede max_length
      # forward the model
      with torch.no_grad(): # don't keep track of gradients in these operations
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # type casting
          logits, loss = model(x, y) # forward pass
        logits = logits[:, -1, :] # get logits at the last position (B, vocab_size)
        probs = F.softmax(logits, dim=-1) # get probabilities (B, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # top-k sampling of 50 (B, 50)

        ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # get token from topk_probs (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # get corresponding indices (B, 1)
        x = torch.cat((x, xcol), dim=1) # append to the sequence

    # print generated text
    for i in range(num_return_sequences):
      tokens = x[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(f"Rank {ddp_rank:02d} | Sample {i:02d} > {decoded}")
      
  # training loop
  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0

  for micro_step in range(gradient_accum_step):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    if ddp:
      model.require_backward_grad_sync = (micro_step == gradient_accum_step - 1)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # type casting
      logits, loss = model(x, y) # forward pass
    loss = loss / gradient_accum_step # loss must be scaled due to gradient accumulation
    loss_accum += loss.detach()
    loss.backward() # backprop

  if ddp:
    dist.allreduce(loss_accum, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # normalize gradients

  # update learning rate using warmup and cosin decay
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  optimizer.step() # update weights

  if torch.cuda.is_available(): torch.cuda.synchronize() # wait for GPU to finish work
  t1 = time.time() # step end time

  dt = t1 - t0 # time difference (in seconds)

  # total token processed in this batch
  tokens_processed = train_loader.B * train_loader.T * gradient_accum_step * ddp_world_size
  tps = tokens_processed / dt # tokens processed per second

  if master_process:
    print(f'step: {step:2d} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tokens/sec: {tps:.2f}')
    with open(log_file, "w") as f:
      f.write(f"{step} TRAIN {loss_accum.item():.4f}\n")

if ddp:
  destroy_process_group()