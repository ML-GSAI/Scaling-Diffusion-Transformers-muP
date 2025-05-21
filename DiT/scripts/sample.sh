torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ # A100-80G or V100-32G
--nnodes=1 \
--node_rank=${NODE_RANK} \
sample_ddp.py \
--load_base_shapes width288_d28.bsh \
--mup \
--num_heads 4 \ # {2, 4, 8}
--ckpt path_ckpt.pth \
--cfg_scale 1 \
--vae mse