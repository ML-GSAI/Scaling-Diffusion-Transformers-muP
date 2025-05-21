cd PixArt-alpha-master/
torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ # A100-80G
--nnodes=4 \
--node_rank=${NODE_RANK} \
train_scripts/train.py \
--config configs/pixart_config/PixArt_mup_xl2_img256_SAM_target.py \
--work-dir output/pretrain_SAM_256_mup/loglr-10 \
--load_base_shapes L28_width288.bsh