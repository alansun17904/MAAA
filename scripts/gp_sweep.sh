# EDGE_SPARSITIES=(0.93 0.94 0.945 0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995 1.0 1.05 1.1 1.15 1.2 1.4)
EDGE_SPARSITIES=(0.93)

for i in "${!EDGE_SPARSITIES[@]}"; do

EDGE_SPARSITY=${EDGE_SPARSITIES[i]}
NODE_SPARSITY=0.69
ELR=0.8
LLR=0.8
RELR=0.8
RLLR=0.8
TOTAL=325 # 3000
WARMUP=75 # 2500

EXTRA="--disable_node_loss"
TAG="wo_node_loss"

# Uncomment this if you want to run with node loss
# EXTRA=""
# TAG="w_node_loss"
VERSION="MAAA" # could be MeZO

train_split="train" # "train_3k"
N_TRAIN=300 # 1000000 # Set to a large value so all of the (150 / 3000) examples are used
N_VAL=150 # The val split size

# You can wrap the following in an sbatch script if you use SLURM
# Activate your environment etc

# If you want to always keep embedding nodes, remove the --with_embedding_nodes flag
# That flag, when set, also models masks over the embedding nodes

WANDB_MODE=disabled python src/layer2/prune/${VERSION}/fpt2_gp.py \
    --report_to wandb \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/gp/ \
    --train_split $train_split \
    --initialize_from gpt2 \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate $ELR \
    --layer_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_layer_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples $N_TRAIN \
    --max_eval_samples $N_VAL \
    --output_dir ./data/runs/gp-${VERSION}-${TAG}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --trainer zo \
    $EXTRA

done