cd PixArt-alpha-master/
torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ # A100-80G
--nnodes=4 \
--node_rank=${NODE_RANK} \
train_scripts/train.py \
--config configs/pixart_config/PixArt_xl2_img256_SAM.py \
--work-dir output/train_SAM_256