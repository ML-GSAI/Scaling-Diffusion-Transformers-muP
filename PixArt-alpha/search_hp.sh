cd PixArt-alpha-master/
torchrun --master_port=${MASTER_PORT} \
--master_addr=${MASTER_ADDR} \
--nproc_per_node=8 \ # A100-80G
--nnodes=1 \
--node_rank=${NODE_RANK} \
train_scripts/train.py \
--config configs/pixart_config/PixArt_mup_xl2_img256_SAM_proxy.py \
--work-dir output/search_SAM_256/loglr-10 \
--load_base_shapes L28_width288.bsh \
--loglr -10 # {-9, -10, -11, -12, -13}