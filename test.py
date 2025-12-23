import os
# This removes the environment variable for the duration of the script
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]

from transformers import AutoModel
import torch

my_token='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'

model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, torch_dtype="auto",token=my_token)
pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, torch_dtype="auto",token=my_token)