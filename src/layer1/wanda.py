import math
from .data import get_loaders 
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

def find_hook_weight(model : HookedTransformer, name, layer_id):
  if "K" in name:
    return ("attn.W_K", model.blocks[layer_id].attn.W_K)
  elif "V" in name:
    return ("attn.W_V", model.blocks[layer_id].attn.W_V)
  elif "Q" in name:
    return ("attn.W_Q", model.blocks[layer_id].attn.W_Q)
  elif "in" in name:
    return ("mlp.W_in", model.blocks[layer_id].mlp.W_in)
  elif "out" in name:
    return ("mlp.W_in", model.blocks[layer_id].mlp.W_out)
  elif "O" in name:
    return ("attn.W_O", model.blocks[layer_id].attn.W_O)
  else:
    return None



def prepare_calibration_input_tlens(model: HookedTransformer, dataloader, max_samples=128):
    all_inps = []
    all_cache = []
    all_tokens = []

    total_samples = 0
    for batch in tqdm(dataloader):
        if total_samples >= max_samples:
            break
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)  # ensure batch dimension

        batch_size = tokens.shape[0]
        hook_names = ["blocks.0.hook_resid_pre"]
        # Run model and cache all activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        # Get input to first transformer block
        # This is the residual stream before layer 0
        resid_pre_0 = cache["resid_pre", 0]  # shape: [batch, seq_len, d_model]

                # Truncate if adding this batch would exceed max_samples
        needed = max_samples - total_samples
        if batch_size > needed:
            resid_pre_0 = resid_pre_0[:needed]
            batch_size = needed

        all_tokens.append(tokens[:batch_size, :])
        all_inps.append(resid_pre_0.cpu())
        total_samples += batch_size

    # Stack inputs
    all_tokens = torch.cat(all_tokens, dim=0) #shape:
    inps = torch.cat(all_inps, dim=0)  # shape: [num_samples, seq_len, d_model]
    outs = torch.zeros_like(inps)      # placeholder (for symmetry with original function)

    return inps, outs, all_tokens

def prune_wanda(args, model, device=torch.device("cuda:0"), sparsity=.5):
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=512, model=model)
    print("dataset loading complete")

    inps, outs, all_tokens = prepare_calibration_input_tlens(model, dataloader)

    inps, outs, all_tokens = inps.to(device), outs.to(device), all_tokens.to(device)

    n_ctx = model.cfg.n_ctx
    n_layers = model.cfg.n_layers
    global_matrix_scores = []  # List of (layer, name, score, weight_pointer)

    weight_names = ["Q", "K", 'V', 'O', 'in', 'out']


                      
    for i in tqdm(range(n_layers)):
        average_cache = {}
        layer = model.blocks[i]
        #Getting TransformerLens names for weights that we want (Q, K, V, O, in, out) and row_scaler of those weights
        for name in weight_names:
            average_cache[name] = {"row_scaler": None,
                                   "count": 0}
        #Getting TransformerLens HookPoints that we want (Hookpoints to get input)
        hook_points = {
            f"blocks.{i}.{name}": module
            for name, module in layer.named_modules()
            if isinstance(module, HookPoint) and ("normalized" in name or "hook_z" in name or "hook_post" in name)
        }
        for j in range(args.nsamples):

            with torch.no_grad():
                for hook_name in hook_points:
                  _, cache = model.run_with_cache(all_tokens[j], names_filter=hook_name)
                  activations = cache[f"{hook_name}"]
                  for name in average_cache:               


                    if average_cache[name]["row_scaler"] is None:
                      average_cache[name]["row_scaler"] = torch.zeros_like(torch.norm(activations, p=2, dim=1))
                    

                    norm = torch.norm(activations, p=2, dim=1)
                    #Take care of the O matrix
                    if norm.ndim == 3:
                        norm = norm.reshape(norm.shape[0], -1)

                    #Matching hook with weight
                    if "ln1.hook_normalized" in hook_name and ('Q'==name or 'K'==name or 'V'==name):
                      average_cache[name]["row_scaler"] = (average_cache[name]["row_scaler"] + norm ** 2) / (average_cache[name]["count"] + 1)
                      average_cache[name]["count"] += 1
                      #Stop at V
                      if average_cache["V"]["row_scaler"] is not None:
                          break
                    elif "ln2.hook_normalized" in hook_name and "in"==name:
                      average_cache[name]["row_scaler"] = (average_cache[name]["row_scaler"] + norm ** 2) / (average_cache[name]["count"] + 1)
                      average_cache[name]["count"] += 1
                      break
                    elif "hook_z" in hook_name and "O"==name:
                      average_cache[name]["row_scaler"] = (average_cache[name]["row_scaler"] + norm ** 2) / (average_cache[name]["count"] + 1)
                      average_cache[name]["count"] += 1
                      break
                    elif "hook_post" in hook_name and "out"==name:#(1, 3072)
                      average_cache[name]["row_scaler"] = (average_cache[name]["row_scaler"] + norm ** 2) / (average_cache[name]["count"] + 1)
                      average_cache[name]["count"] += 1
                      break
        for name in average_cache:
            weight_info = find_hook_weight(model, name, i)
            W = weight_info[1]

            row_scaler = average_cache[name]["row_scaler"]

            '''
            print(f"Weight shape = {W.shape}")
            print(f"Row scaler shape = {row_scaler.shape}")'''
            W_metric = torch.abs(W.view((-1, row_scaler.shape[1]))) * torch.sqrt(row_scaler)
            score = torch.sum(W_metric).item()  # Single scalar for the entire matrix
            global_matrix_scores.append({
                "name":weight_info[0],
                "layer": i,
                "score": score,
                "weight": W  # In-place modifiable tensor
            })

    # Sort by score (descending): higher scores = more important
    global_matrix_scores.sort(key=lambda x: x["score"], reverse=True)

    # Determine number to keep
    num_total = len(global_matrix_scores)
    num_keep = int((1 - sparsity) * num_total)

    # Prune the rest (set entire weight matrix to zero)
    for idx in range(num_keep, num_total):
        with torch.no_grad():
            global_matrix_scores[idx]["weight"].zero_()  # In-place zeroing

    return global_matrix_scores