# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import transformer_lens as tl
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, model):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = model.to_tokens(" ".join(traindata['text']))
    testenc = model.to_tokens("\n\n".join(testdata['text']))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, model):
    # Load train and validation datasets
    print("Loading train dataset...")
    traindata = load_dataset('allenai/c4', data_files='en/c4-train.00000-of-01024.json.gz', split = "train")
    print(f"Loaded validation dataset with {len(traindata)} samples")
    print("Loading validation dataset...")
    valdata = load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split = "validation")
    print(f"Loaded validation dataset with {len(valdata)} samples")


    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = model.to_tokens(traindata[i]['text'])
            if trainenc.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = model.to_tokens(' '.join(valdata[:1100]['text']))

    valenc = valenc[:, :(256 * seqlen)]
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=512, model=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model)