from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from functools import partial


def load_dataset(path, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, device):
    # In the future, we can refactor this to account for the various different types of datasets or smth
    dataset = load_dataset('csv', data_files=path)
    dataset.set_format(type='torch')

    dataset = dataset.map(partial(generate_data, model, tokenizer, device), batched=True)
    dataset = dataset.map()
    return dataset


def generate_data(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, device, example):
    tokens = tokenizer(example['corrupted'], return_tensors='pt').to(device)
    with torch.no_grad():
        example['corr_logits'] = model(**tokens, max_new_tokens=10).logits
    # Fix device & max new tokens
    return example