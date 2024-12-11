import sys
import logging
import datasets
from datasets import load_dataset
acc_dataset = load_dataset("lambada", split="validation[:40]")
perp_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
