import os 
from easydict import EasyDict 
from datetime import datetime 

#------------------------ environment ------------------------#
cfg = EasyDict(__name__='Config: VAE')
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

#------------------------ data ------------------------#
# sampler
cfg.root = 'datasets/imagenet/train' 
cfg.image_list_path = 'imagenet_train.txt'
cfg.img_size = 256
cfg.batch_size = 48
cfg.seed = 1

# dataloader
cfg.num_workers = 12
cfg.prefetch_factor = 2

#------------------------ model ------------------------#
# vae
cfg.model_load_from  = None
cfg.codebook_size = 4096 
cfg.token_size = 32 
cfg.patch_size = 16 
cfg.aux_tokens_num = 17 # prefix_token
cfg.stage = 'stage1'  # stage1 or stage2
cfg.clustering_vq = False # True for stage1

#------------------------ training ------------------------#
# criterion  
cfg.disc_start = 1_000_000 
cfg.perceptual_weight = 1
cfg.lpips_weight = 0

cfg.reconstruction_weight = 1
cfg.quantizer_weight = 1
cfg.discriminator_factor = 0.1
cfg.discriminator_weight = 0.05

# optimizer
cfg.num_steps = 1_000_000
cfg.warmup_steps = 10_000
cfg.lr = 1e-4

# logging
cfg.log_interval = 500 # print log and visualization
cfg.last_checkpoint_interval = 1000 # save_last_checkpoint
cfg.checkpoint_interval = 5000  # save_checkpoint

cfg.log_dir = 'vae_{}'.format(
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)   
