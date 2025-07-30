import math
from .data import get_loaders 
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

def find_hook_weight(model : HookedTransformer, name, layer_id):

  #For GroupedQueryAttention  
  if "_V" in name:
    return model.blocks[layer_id].attn._W_V
  elif "_K" in name:
    return model.blocks[layer_id].attn._W_K
  elif "K" in name:
    return model.blocks[layer_id].attn.W_K
  elif "V" in name:
    return model.blocks[layer_id].attn.W_V
  elif "Q" in name:
    return model.blocks[layer_id].attn.W_Q
  elif "in" in name:
    return model.blocks[layer_id].mlp.W_in
  elif "out" in name:
    return model.blocks[layer_id].mlp.W_out
  elif "O" in name:
    return model.blocks[layer_id].attn.W_O
  elif "gate" in name:
    return model.blocks[layer_id].mlp.W_gate
  else:
    return None



def prepare_calibration_input_tlens(model: HookedTransformer, dataloader, max_samples=128):
    all_inps = []
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

def prune_wanda(args, model, device=torch.device("cuda:0"), sparsity=0):
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=512, model=model)
    print("dataset loading complete")

    inps, outs, all_tokens = prepare_calibration_input_tlens(model, dataloader, max_samples=args.nsamples)

    inps, outs, all_tokens = inps.to(device), outs.to(device), all_tokens.to(device)

    model = model.to(device)

    n_layers = model.cfg.n_layers
    global_matrix_scores = {}  # List of {layer, name, head_idx, score, mask}

    
    weight_names = []
    for name, _ in model.named_parameters():
        if "W_E" in name:
            continue
        elif "1" in name:
            break
        elif "W" in name:           
            split_name = name.split(".")
            weight_names.append(split_name[-2] + "." + split_name[-1])
        
                      
    for i in tqdm(range(n_layers)):
        average_cache = {}
        layer = model.blocks[i]
        #Getting TransformerLens names for weights that we want (Q, K, V, O, in, out) and row_scaler of those weights
        for name in weight_names:
            average_cache[name] = {"row_scaler": None,
                                   "count": 0}
        #Getting TransformerLens HookPoints that we want (Hookpoints to get input)

        hook_points = [ #Modify this so I directly get the inputs to each weight
            f"blocks.{i}.ln1.hook_normalized",  # Input to Q, K, V
            f"blocks.{i}.ln2.hook_normalized",  # Input to MLP (Need for return)
            f"blocks.{i}.attn.hook_z",         # Input to O matrix
            f"blocks.{i}.mlp.hook_post",       # For out matrix
        ]
        for j in range(args.nsamples):
            
            with torch.no_grad():
                _, cache = model.run_with_cache(all_tokens[j], names_filter=hook_points) #Has to be in here cuz too much memory outside of loop
                for hook_name in hook_points:       
                    
                    activations = cache[f"{hook_name}"]
                    #print(f"{hook_name}: {activations.shape}")
                    
                    #Flatten across all but last dimension
                    dims_to_reduce = tuple(range(activations.ndim - 1))
                    norm = torch.norm(activations, p=2, dim=dims_to_reduce)

                    
                    if "ln1.hook_normalized" in hook_name:
                        # This feeds into Q, K, V matrices
                        for name in ["attn.W_Q", "attn.W_K", "attn.W_V"]:
                            #Take care of GroupedQueryAttention
                            if name not in weight_names:
                                name = name.split(".")[-2] + "._" + name.split(".")[-1]
                            if average_cache[name]["row_scaler"] is None:
                                average_cache[name]["row_scaler"] = torch.zeros(norm.shape[-1]).to(device)
                            average_cache[name]["row_scaler"] = (average_cache[name]["row_scaler"] * average_cache[name]["count"] + norm ** 2) / (average_cache[name]["count"] + 1)
                            average_cache[name]["count"] += 1
                    elif "hook_z" in hook_name:
                        # This feeds into O matrix
                        if average_cache["attn.W_O"]["row_scaler"] is None:
                            average_cache["attn.W_O"]["row_scaler"] = torch.zeros(norm.shape[-1]).to(device)
                        average_cache["attn.W_O"]["row_scaler"] = (average_cache["attn.W_O"]["row_scaler"] * average_cache["attn.W_O"]["count"] + norm ** 2) / (average_cache["attn.W_O"]["count"] + 1)
                        average_cache["attn.W_O"]["count"] += 1
    
                    elif "ln2.hook_normalized" in hook_name:
                        # This feeds into MLP input matrix
                        if average_cache["mlp.W_in"]["row_scaler"] is None:
                            average_cache["mlp.W_in"]["row_scaler"] = torch.zeros(norm.shape[-1]).to(device)
                        average_cache["mlp.W_in"]["row_scaler"] = (average_cache["mlp.W_in"]["row_scaler"] * average_cache["mlp.W_in"]["count"] + norm ** 2) / (average_cache["mlp.W_in"]["count"] + 1)
                        average_cache["mlp.W_in"]["count"] += 1
                        if "mlp.W_gate" in weight_names:
                            if average_cache["mlp.W_gate"]["row_scaler"] is None:
                                average_cache["mlp.W_gate"]["row_scaler"] = torch.zeros(norm.shape[-1]).to(device)
                            average_cache["mlp.W_gate"]["row_scaler"] = (average_cache["mlp.W_gate"]["row_scaler"] * average_cache["mlp.W_gate"]["count"] + norm ** 2) / (average_cache["mlp.W_gate"]["count"] + 1)
                            average_cache["mlp.W_gate"]["count"] += 1
 
                    elif "hook_post" in hook_name or "hook_mid" in hook_name:
                        # This is for MLP output matrix
                        if average_cache["mlp.W_out"]["row_scaler"] is None:
                            average_cache["mlp.W_out"]["row_scaler"] = torch.zeros(norm.shape[-1]).to(device)
                        average_cache["mlp.W_out"]["row_scaler"] = (average_cache["mlp.W_out"]["row_scaler"] * average_cache["mlp.W_out"]["count"] + norm ** 2) / (average_cache["mlp.W_out"]["count"] + 1)
                        average_cache["mlp.W_out"]["count"] += 1

        
        layer_matrix_scores = []
        
        for name in average_cache:
            W = find_hook_weight(model, name, i)

            row_scaler = torch.sqrt(average_cache[name]["row_scaler"])
            #print(name)
                        # Handle dimension matching
            if W.dim() == 2:  # Standard weight matrix
                if row_scaler.shape[0] != W.shape[0]:
                    print(f"Warning: Dimension mismatch for {name} at layer {i}")
                    print(f"Weight shape: {W.shape}, Row scaler shape: {row_scaler.shape}")
                    continue
                    
                W_metric = torch.abs(W) * row_scaler.view(-1, 1)
            elif W.dim() == 3:  # Multi-head attention weights (LLaMA)
                # Reshape to handle multi-head structure
                if row_scaler.shape[0] != W.shape[1]:
                    print(f"Warning: Dimension mismatch for {name} at layer {i}")
                    print(f"Weight shape: {W.shape}, Row scaler shape: {row_scaler.shape}") 
                    continue
                #print(W.shape)
                W_metric = torch.abs(W) * row_scaler.view(1, -1, 1)
            else:
                print(f"Unexpected weight dimension for {name}: {W.shape}")
                continue
    
            
            #Prune by each head
            if "mlp" not in name:
                for head_idx in range(W_metric.shape[0]):                
                    score = torch.sum(W_metric[head_idx]).item()  # Single scalar for the entire matrix
                    global_matrix_scores[f"block.{i}.{name}[{head_idx}]"] = {
#                        "head_idx": head_idx, #For right now remove
                        "score": score,  # In-place modifiable tensor
                        "mask": 1
                    }
                    layer_matrix_scores.append({
                        "name":name,
                        "head_idx": head_idx,
                        "score": score  # In-place modifiable tensor
                    })
                
            else: 
                score = torch.sum(W_metric).item()  # Single scalar for the entire matrix
                global_matrix_scores[f"block.{i}.{name}"] = {
                    "score": score,  # In-place modifiable tensor
                    "mask": 1
                }
        layer_matrix_scores.sort(key=lambda x: x["score"], reverse=False)
        print(layer_matrix_scores)
        # Determine number to keep
        num_total = len(layer_matrix_scores)
        num_keep = int((1-sparsity) * num_total)
        # Prune the rest (set entire weight matrix to zero)
        for idx in range(num_keep, num_total):
            with torch.no_grad():
                name = layer_matrix_scores[idx]["name"]
                layer = i
                W = find_hook_weight(model, name, layer)
                
                if "mlp" not in name:
                    head_idx = layer_matrix_scores[idx]["head_idx"]
                    W[head_idx] = 0 # In-place zeroing
                    global_matrix_scores[f"block.{i}.{name}[{head_idx}]"]["mask"] = 0
                else:
                    continue        


    # Sort by score (descending): higher scores = more important
    #global_matrix_scores = dict(sorted(global_matrix_scores.items(), key=lambda item: item[1]["score"], reverse=True))
    

    return global_matrix_scores