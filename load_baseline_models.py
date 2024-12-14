import argparse
from huggingface_hub import login
import os
import torch
import tqdm
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, LlamaForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import WQAQLinear, quantize_opt


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='facebook/opt-125m')
args = parser.parse_args()

login(token=os.environ["HF_TOKEN"])

model_name = args.model_name
opt_model_fp16 = OPTForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

opt_model_fp16_2 = OPTForCausalLM.from_pretrained(
    "facebook/opt-6.7b", torch_dtype=torch.float16, device_map="auto"
)
opt_model_fp16_3 = OPTForCausalLM.from_pretrained(
    "facebook/opt-13b", torch_dtype=torch.float16, device_map="auto"
)
llama_2_7b_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto"
)
