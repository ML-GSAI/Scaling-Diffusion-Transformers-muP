_base_ = ['../PixArt_xl2_sam.py']
data_root = 'data'
image_list_txt = ['part00.txt', 'part01.txt', 'part02.txt', 'part03.txt', 'part04.txt', 'part05.txt', 'part06.txt', 'part07.txt', 'part08.txt',
                  'part09.txt', 'part10.txt', 'part11.txt', 'part12.txt', 'part13.txt', 'part14.txt','part15.txt','part16.txt',
                  'part17.txt','part18.txt','part19.txt','part20.txt','part21.txt', 'part22.txt', 'part23.txt', 'part24.txt',
                  'part25.txt', 'part26.txt', 'part27.txt', 'part28.txt', 'part29.txt', 'part30.txt', 'part31.txt']
data = dict(type='SAM', root='SA1B', image_list_txt=image_list_txt, transform='default_train', load_vae_feat=False)
image_size = 256

# model setting
mup=True
head=16

window_block_indexes=[]
window_size=0
use_rel_pos=False
model = 'PixArt_ANY'
fp32_attention = True
load_from = None
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10
train_batch_size = 176
num_epochs = 30
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='MuAdamW', lr=2**-10, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
log_interval = 20
save_model_epochs=2
save_model_steps=20000
