from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from functools import partial


def ca_load_dataset(path, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, device, max_new_tokens, batch_size):
    # In the future, we can refactor this to account for the various different types of datasets or smth
    dataset = load_dataset('csv', data_files=path)
    dataset.set_format(type='torch')

    dataset = dataset.map(partial(generate_data, model, tokenizer, device, max_new_tokens), batched=True, batch_size=batch_size)
    dataset = dataset.map() # yo wtf
    return dataset


def generate_data(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, device, tokens, example):
    tokens = tokenizer(example['corrupted'], return_tensors='pt', padding=True, padding_side='left').to(device)
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=tokens, return_dict_in_generate=True, output_logits=True)
        example['corr_logits'] = torch.transpose(torch.stack(out.logits), 0, 1)
    # Fix device & max new tokens
    return example