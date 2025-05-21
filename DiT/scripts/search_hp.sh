torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ #A100-80G GPU
--nnodes=1 \
--node_rank=${NODE_RANK} \
train_mup.py \
--load_base_shapes width288_d28.bsh \
--mup \
--global_batch_size 256 \
--num_heads 4 \ # {2, 4, 8}
--epochs 40 \
--loglr -10 # {-9, -10, -11, -12, -13}