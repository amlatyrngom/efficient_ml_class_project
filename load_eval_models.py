import argparse
from huggingface_hub import login
import os
from transformers import GPT2Tokenizer, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='facebook/opt-125m')
args = parser.parse_args()
perp_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
acc_tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
perp_tokenizer_2 = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
acc_tokenizer_2 = GPT2Tokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
perp_tokenizer_3 = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)
acc_tokenizer_3 = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)

login(token=os.environ["HF_TOKEN"])
perp_tokenizer_llama = tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
