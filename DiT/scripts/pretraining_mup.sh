torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ # A100-80G
--nnodes=4 \
--node_rank=${NODE_RANK} \
train_mup.py \
--load_base_shapes width288_d28.bsh \
--mup \
--global_batch_size 256 \
--num_heads 16 \
--epochs 480 \
--loglr -10