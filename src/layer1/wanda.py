import math
from .data import get_loaders 
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

def find_hook_weight(model : HookedTransformer, name, layer_id):

  #For GroupedQueryAttention  
  if "_W_V" in name:
    return model.blocks[layer_id].attn._W_V
  elif "_W_K" in name:
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



def prepare_calibration_input_tlens(model: HookedTransformer, dataloader, corrdataloader, seqlen=25, max_samples=128, device = torch.device("cuda:0")):
    all_inps = []
    all_cache = []
    all_tokens = []
    all_corr_tokens = []

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
        resid_pre_0 = cache["resid_pre", 0]# shape: [batch, seq_len, d_model]

                # Truncate if adding this batch would exceed max_samples
        needed = max_samples - total_samples
        if batch_size > needed:
            resid_pre_0 = resid_pre_0[:needed]
            batch_size = needed
        if tokens.shape[1] < seqlen:
            # Pad to the right
            pad_len = seqlen - tokens.shape[1]
            tokens = torch.cat([tokens[:batch_size, :], torch.zeros(1, pad_len, dtype=tokens.dtype).to(device)], dim=1)
            resid_pre_0 = torch.cat([resid_pre_0, torch.zeros(1, pad_len, resid_pre_0.shape[2], dtype=resid_pre_0.dtype).to(device)], dim=1)
        all_tokens.append(tokens[:batch_size, :])
        all_inps.append(resid_pre_0.cpu())
        total_samples += batch_size
    
    
    total_samples = 0
    for batch in tqdm(corrdataloader):
        if total_samples >= max_samples:
            break
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)  # ensure batch dimension
        if tokens.shape[1] < seqlen:
            # Pad to the right
            pad_len = seqlen - tokens.shape[1]
            tokens = torch.cat([tokens[:batch_size, :], torch.zeros(1, pad_len, dtype=tokens.dtype).to(device)], dim=1)
        batch_size = tokens.shape[0]
        all_corr_tokens.append(tokens)
        total_samples += batch_size
    # Stack inputs
    
    all_tokens = torch.cat(all_tokens, dim=0) #shape: [num_samples, seq_len, d_model]
    all_corr_tokens = torch.cat(all_corr_tokens, dim=0)
    inps = torch.cat(all_inps, dim=0)  # shape: [num_samples, seq_len, d_model]
    outs = torch.zeros_like(inps)      # placeholder (for symmetry with original function)

    return inps, outs, all_tokens, all_corr_tokens

def prune_wanda(args, model, device=torch.device("cuda:0"), sparsity=0):
    print("loading calibdation data")
    dataloader, _, corrdataloader, _ = get_loaders("ioi",nsamples=args.prune_nsamples,seed=args.prune_seed,seqlen=25, model=model)
    print("dataset loading complete")

    inps, outs, all_tokens, all_corr_tokens = prepare_calibration_input_tlens(model, dataloader, corrdataloader, seqlen=25, max_samples=args.prune_nsamples, device = device)

    inps, outs, all_tokens, all_corr_tokens = inps.to(device), outs.to(device), all_tokens.to(device), all_corr_tokens.to(device)

    model = model.to(device)

    n_layers = model.cfg.n_layers
    global_matrix_scores = []  # List of {layer, name, head_idx, score, mask}
    

    q_min = 0
    k_min = 0
    v_min = 0
    o_min = 0
    
    q_max = 0
    k_max = 0
    v_max = 0
    o_max = 0

    
    q_matrix_scores = []
    k_matrix_scores = []
    v_matrix_scores = []
    o_matrix_scores = []
    
    q_scores = []
    k_scores = []
    v_scores = []
    o_scores = []
    
    mlps = []
    weight_names = []
    for name, _ in model.named_parameters():
        if "W_E" in name or "embed" in name:
            continue
        elif "1" in name:
            break
        elif "W" in name:           
            split_name = name.split(".")
            weight_names.append(split_name[-2] + "." + split_name[-1])
        
                      
    for i in tqdm(range(n_layers)):
        average_cache = {}
        #Getting TransformerLens names for weights that we want (Q, K, V, O, in, out) and row_scaler of those weights
        for name in weight_names:
            average_cache[name] = {"row_scaler": None,
                                   "count": 0}
        #Getting TransformerLens HookPoints that we want (Hookpoints to get input)

        hook_points = [ #Modify this so I directly get the inputs to each weight
            f"blocks.{i}.ln1.hook_normalized",  # Input to Q, K, V
            f"blocks.{i}.ln2.hook_normalized",  # Input to MLP (Need for return)
            f"blocks.{i}.attn.hook_z",         # Input to O matrix
            f"blocks.{i}.mlp.hook_post",       # Output of MLP (for out matrix)
        ]
        for j in range(args.prune_nsamples):
            
            with torch.no_grad():
                _, cache = model.run_with_cache(all_tokens[j], names_filter=hook_points) #Has to be in here cuz too much memory outside of loop
                _, corrcache = model.run_with_cache(all_corr_tokens[j], names_filter=hook_points)
                for hook_name in hook_points:       
                    
                    activations = cache[f"{hook_name}"]
                    corractivations = corrcache[f"{hook_name}"]
                    #print(f"{hook_name}: {activations.shape}")
                    dims_to_reduce = tuple(range(activations.ndim - 1))
                    activations = (activations - activations.mean()) / (activations.std())
                    corractivations = (corractivations - corractivations.mean()) / (corractivations.std())
                    norm = torch.norm((corractivations-activations), p=2, dim=dims_to_reduce) 
                    
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

        
        
        for name in average_cache:
            W = find_hook_weight(model, name, i)
                
            row_scaler = torch.sqrt(average_cache[name]["row_scaler"])
            row_scaler = row_scaler.mean()

            W_metric = torch.abs(W) * row_scaler
            '''
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
            '''
            #Prune by each head
            if "mlp" not in name:
                for head_idx in range(W_metric.shape[0]):                
                    score = W_metric[head_idx].mean().item()  # Single scalar for the entire matrix
                    info = {
                            "name": f"block.{i}.{name}",
                            "head_idx": head_idx, #For right now remove
                            "score": score,  # In-place modifiable tensor
                            "mask": 1
                        }
                    if "W_Q" in name:
                        q_matrix_scores.append(info)
                        q_scores.append(score)
                        if score > q_max:
                            q_max = score
                        elif score < q_min:
                            q_min = score
                    elif "W_K" in name:
                        k_matrix_scores.append(info)
                        k_scores.append(score)
                        if score > k_max:
                            k_max = score
                        elif score < k_min:
                            k_min = score
                    elif "W_V" in name:
                        v_matrix_scores.append(info)
                        v_scores.append(score)
                        if score > v_max:
                            v_max = score
                        elif score < v_min:
                            v_min = score
                                
                    elif "W_O" in name:
                        o_matrix_scores.append(info)
                        o_scores.append(score)
                        if score > o_max:
                            o_max = score
                        elif score < o_min:
                            o_min = score
            else:
                info = {
                            "name": f"block.{i}.{name}",
                            "mask": 1
                        }
                mlps.append(info)
    
    q_std = torch.tensor(q_scores).std()
    k_std = torch.tensor(k_scores).std()
    v_std = torch.tensor(v_scores).std()
    o_std = torch.tensor(o_scores).std()

    q_avg = torch.tensor(q_scores).mean()
    k_avg = torch.tensor(k_scores).mean()
    v_avg = torch.tensor(v_scores).mean()
    o_avg = torch.tensor(o_scores).mean()
    '''
    for info in q_matrix_scores:
        info["score"] = (info["score"] - q_min) / (q_max - q_min)
    for info in k_matrix_scores:
        info["score"] = (info["score"] - k_min) / (k_max - k_min)
    for info in v_matrix_scores:
        info["score"] = (info["score"] - v_min) / (v_max - v_min)
    for info in o_matrix_scores:
        info["score"] = (info["score"] - o_min) / (o_max - o_min)
    '''
    for info in q_matrix_scores:
        info["score"] = (info["score"] - q_avg) / (q_std)
    for info in k_matrix_scores:
        info["score"] = (info["score"] - k_avg) / (k_std)
    for info in v_matrix_scores:
        info["score"] = (info["score"] - v_avg) / (v_std)
    for info in o_matrix_scores:
        info["score"] = (info["score"] - o_avg) / (o_std)
    
    
    global_matrix_scores.extend(q_matrix_scores)
    global_matrix_scores.extend(k_matrix_scores)
    global_matrix_scores.extend(v_matrix_scores)
    global_matrix_scores.extend(o_matrix_scores)
    
    q_matrix_scores.clear()
    k_matrix_scores.clear()
    v_matrix_scores.clear()
    o_matrix_scores.clear()   


    global_matrix_scores.sort(key=lambda x: x["score"], reverse=True)
    # Determine number to keep
    num_total = len(global_matrix_scores)
    num_keep = int((1-sparsity) * num_total)
    # Prune the rest (set entire weight matrix to zero)
    for idx in range(num_keep, num_total):
        with torch.no_grad():
            name = global_matrix_scores[idx]["name"]
            layer = int(name.split(".")[1])
            W = find_hook_weight(model, name, layer)
            
            if "mlp" not in name:
                head_idx = global_matrix_scores[idx]["head_idx"]
                W[head_idx] = 0 # In-place zeroing
                global_matrix_scores[idx]["mask"] = 0
            else:
                continue
#                    W.zero_()  # Skip mlp weights
#                    global_matrix_scores[f"block.{i}.{name}"]["mask"] = 0               


    # Sort by score (descending): higher scores = more important
#    global_matrix_scores = dict(sorted(global_matrix_scores.items(), key=lambda item: item[1]["score"], reverse=True))

    global_matrix_scores.extend(mlps)
    global_matrix_scores.extend([{"name" : "embed", "mask" : 1}, {"name" : "unembed", "mask" : 1}])
    return global_matrix_scores