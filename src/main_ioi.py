from layer1.wanda import prune_wanda
from layer1.label_smoothing_utils import compute_z_star, compute_edge_scores
from layer2.prune.edge_pruning_mezo.fpt2_ioi import load_datasets, freeze_all_except_pruning_params, DataCollatorIOI, get_optimizers, FPT2InfoTrainer, eval_fn, ModelArguments, DataTrainingArguments, MeZOTrainingArguments
from layer2.modeling.modeling_fpt2 import FPT2LMHeadModel
import logging as logging_py
import os
import torch
import pickle
import random
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional
from transformer_lens import HookedTransformer

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW

import sys
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging_py.getLogger(__name__)

@dataclass
class PruningArguments:
    prune_nsamples: Optional[int] = field(
        default=128,
        metadata={"help": "Number of samples to use for pruning. If None, use all samples."},
    ),
    prune_seed: Optional[int] = field(
        default=123,
        metadata={"help": "Random seed for selecting samples for pruning."},
    ),
    prune_sparsity: Optional[float] = field(
        default=0.86,
        metadata={"help": "Target sparsity for pruning."},
    ),


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda:0", center_writing_weights=False)


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MeZOTrainingArguments, PruningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, prune_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, prune_args = parser.parse_args_into_dataclasses()

    prune_args.prune_seed = training_args.seed
    w = prune_wanda(prune_args, model, sparsity=prune_args.prune_sparsity)
    w = compute_z_star(w, alpha=0.1)
    reading_scores, writing_scores = compute_edge_scores(w)


    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging_py.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging_py.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples, train_split=data_args.train_split)
    n_train = len(raw_datasets["train"])
    
    model = FPT2LMHeadModel.from_pretrained(
        model_args.initialize_from,
        with_embedding_nodes=data_args.with_embedding_nodes,
        disable_linear_regularization_term=data_args.disable_linear_reg_term,
        reading_scores = reading_scores,
        writing_scores = writing_scores,
    )
    gpt2_model = FPT2LMHeadModel.from_pretrained(
        "gpt2",
        with_embedding_nodes=data_args.with_embedding_nodes,
        reading_scores = reading_scores,
        writing_scores = writing_scores,
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    freeze_all_except_pruning_params(model)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    # Data collator
    collator = DataCollatorIOI(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
    
    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=data_args.disable_node_loss
    )

    # Initialize our Trainer
    trainer = FPT2InfoTrainer(
        model=model,
        gpt2_model=gpt2_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_layer_sparsity=data_args.start_layer_sparsity,
        target_layer_sparsity=data_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=data_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": "gpt-2"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()