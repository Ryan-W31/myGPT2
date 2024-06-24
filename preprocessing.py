# preprocessing.py - myGPT2 - Ryan-W31
"""
Downloads and tokenizes the FineWeb-edu 10B dataset and saves data shards to disk.

FineWeb-edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Run Command:
python preprocessing.py

Will save shard to local directory "fineweb_edu_10B"
"""

# Package Imports
# -----------------------------------------------------
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
# -----------------------------------------------------

# Global Variables
# -----------------------------------------------------
ld = "fineweb_edu_10B" # local directory
rd = "sample-10BT" # remote directory
shard_size = int(1e8) # each shard will have 100M tokens
# -----------------------------------------------------

# Processing
# -----------------------------------------------------

# create data cache to local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ld)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=rd, split="train")

# initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # special end-of-text token

# tokenizes a single document and returns a numpy array of uint16 tokens
def tokenize(data):
  tokens = [eot] # start tokens with special token
  tokens.extend(enc.encode_ordinary(data['text'])) # appends all tokens in document to tokens
  tokens_np = np.array(tokens) # convert to numpy array

  assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
  tokens_np_uint16 = tokens_np.astype(np.uint16) # convert tokens to uint16

  return tokens_np_uint16

# write tokens to file as bytes
def write_to_file(filename, tokens_np):
  np.save(filename, tokens_np)

num_processes = max(1, os.cpu_count() // 2) # use multiprocessing
with mp.Pool(num_processes) as pool:
  shard_index = 0 # current shard number
  all_tokens_np = np.empty((shard_size, ), dtype=np.uint16) # pre-allocate buffer to hold all tokens
  token_count = 0
  progress_bar = None

  for tokens in pool.imap(tokenize, fw, chunksize=16):
    if token_count + len(tokens) < shard_size: # determine if there is space in the shard for new tokens

      # append new tokens to shard
      all_tokens_np[token_count : token_count + len(tokens)] = tokens
      token_count += len(tokens)

      # update progress bar
      if progress_bar is None:
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
      progress_bar.update(len(tokens))

    else: # otherwise

      # write the current shard to file and start a new shard
      split = "val" if shard_index == 0 else "train"
      filename = os.path.join(DATA_CACHE_DIR, f"fwedu_{split}_{shard_index:04d}.npy")

      # split document into what goes into the shard and the remainder
      remainder = shard_size - token_count
      progress_bar.update(remainder)
      all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]

      write_to_file(filename, all_tokens_np)
      shard_index += 1
      progress_bar = None

      # populate the next shard with the remainder
      all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
      token_count = len(tokens) - remainder

  # write any remaining tokens to the last shard
  if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"fwedu_{split}_{shard_index:04d}.npy")
    write_to_file(filename, all_tokens_np[:token_count])
# -----------------------------------------------------


