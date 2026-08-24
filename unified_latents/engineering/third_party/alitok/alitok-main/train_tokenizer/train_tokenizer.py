import os
import os.path as osp
import sys
import torch 
torch.set_warn_always(False)
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
import datetime
import logging  
from torch.nn.parallel import DistributedDataParallel as DDP 
from importlib import reload 
import tok_utils
from config import cfg 
from loss.mse_lpips_gan import MSE_LPIPS_GAN
from torch.optim.lr_scheduler import LambdaLR   
from tok_utils import ImageDataset, LambdaLinearScheduler, AverageMeter
import random
import numpy as np
import argparse

def main(args):
    cfg.update(args) 
    cfg.pmi_rank = int(os.environ['RANK'])
    cfg.pmi_world_size = int(os.environ['WORLD_SIZE'])
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def worker(gpu, cfg):  
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    # init distributed processes
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    dist.init_process_group(
        backend='nccl', world_size=cfg.world_size, rank=cfg.rank,
        timeout=datetime.timedelta(hours=5)
    )

    # logging
    reload(logging)
    
    restart_time = 0 
    start_step = 0 
    # load the information of the last checkpoint
    if os.path.exists('last.pth'):
        checkpoint = torch.load('last.pth', map_location='cpu')             
        cfg.model_load_from ='last.pth' 
        if 'restart_time'in checkpoint:
            restart_time = checkpoint['restart_time'] + 1
            cfg.seed += restart_time * 1000 
        cfg.log_dir = checkpoint['log_dir']
        start_step = checkpoint['start_step'] 
        del checkpoint
    
    cfg.log_dir = tok_utils.generalized_all_gather(cfg.log_dir)[0]
    if cfg.rank == 0:
        log_key = os.path.join(
            cfg.log_dir,
            f'{cfg.log_dir}_restart{restart_time}.log' 
        )
        log_cth = f'logs/{log_key}'
        os.makedirs(osp.dirname(log_cth), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_cth),
                logging.StreamHandler(stream=sys.stdout)
            ])
        
        # initial logging
        logging.info(f'Logging to {log_key}')
        logging.info(cfg)
        
    # seed 
    seed = cfg.seed + cfg.rank
    random.seed(seed)     
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    # data
    dataset = ImageDataset(cfg.img_size, cfg.root, cfg.image_list_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, sampler=sampler,
                     num_workers=cfg.num_workers, pin_memory=True, drop_last=True, generator=dataloader_generator)
    rank_iter = iter(dataloader)

    # model
    if cfg.stage == 'stage1':
        from model.vae_stage1 import AliTok_Stage1
        model = AliTok_Stage1(cfg).to(gpu) 
        optimizer = optim.AdamW(
            params=model.parameters(), 
            lr=cfg.lr)    
        
    elif cfg.stage == 'stage2':
        from model.vae_stage2 import AliTok_Stage2
        model = AliTok_Stage2(cfg).to(gpu) 
        params_to_optimize = []
        for name, param in model.named_parameters():         
            if  'encoder' in name or 'quantize' in name  or 'latent_tokens' in name :  
                param.requires_grad = False 
            else:
                params_to_optimize.append(param)    
            
        optimizer = optim.AdamW(
            params=params_to_optimize, 
            lr=cfg.lr)    
    
    criterion = MSE_LPIPS_GAN(cfg).to(gpu)
    
    optimizer_d = optim.AdamW(
        params=list(criterion.discriminator.parameters()),
        lr=cfg.lr)  
    scaler = torch.cuda.amp.GradScaler(enabled=True) 

    # warmup
    warm_up_steps = max(0, cfg.warmup_steps-start_step)
    f_start = min(1, start_step / cfg.warmup_steps)
    
    lr_scheduler = LambdaLinearScheduler(warm_up_steps=[ warm_up_steps ], f_min=[ 1. ], f_max=[ 1. ], \
         f_start=[ f_start ], cycle_lengths=[ 10000000000000 ])
    lr_scheduler.last_f = start_step-1
    scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler) 
    
    # load_last_weight
    if cfg.model_load_from:
        checkpoint = torch.load(cfg.model_load_from, map_location='cpu') 
        if 'criterion_state'  in checkpoint: 
            tok_utils.load_state_dict(criterion, checkpoint["criterion_state"])
        if 'scaler_state' in checkpoint: 
            scaler.load_state_dict(checkpoint["scaler_state"])
        else:
            scaler_state = scaler.state_dict() 
            scaler_state["scale"] = 1.0 
            scaler.load_state_dict(scaler_state)
        if 'model_state' in checkpoint:
            tok_utils.load_state_dict(model, checkpoint['model_state'])
        else:
            tok_utils.load_state_dict(model, checkpoint)
        del checkpoint
        
    optimizer.zero_grad() 
    optimizer_d.zero_grad()
    
    criterion = DDP(criterion, find_unused_parameters=True)  
    if cfg.stage == 'stage1':
        model = DDP(model)
    elif cfg.stage == 'stage2':
        model = DDP(model, find_unused_parameters=True)
        
    # empty cache before training
    torch.cuda.empty_cache()
    avg_loss_vq = AverageMeter()
    avg_loss_mse = AverageMeter()
    avg_loss_lpips = AverageMeter()
    avg_loss_gan = AverageMeter()
    avg_loss_discr = AverageMeter()
    avg_logit_real = AverageMeter()
    avg_logit_fake = AverageMeter() 
    
    # training loop  
    total_step = start_step
    optimizer_idx = 0 
    codebook_size = cfg.codebook_size 
    for step in range(total_step, cfg.num_steps + 1):   
        total_step += 1
        videos = next(rank_iter).to(gpu)
        # forward 
        with amp.autocast(): 
            recons, x_aux, extra_results_dict, codebook_idx  = model(videos)
        recons = recons.float() 
        if x_aux != None:
            x_aux = x_aux.float()
        
        if optimizer_idx == 0:   
            mse, lpips, vq, gan = criterion(
                videos,
                recons,
                x_aux, 
                extra_results_dict,   
                total_step,
                "generator"
            ) 
            if cfg.stage == 'stage1':
                loss = mse + lpips + vq + gan   
            elif cfg.stage == 'stage2':
                loss = mse + lpips + gan 
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward() 
            scaler.step(optimizer)
            scaler.update() 
            scheduler.step()
                                    
            metrics_mse = torch.stack([mse]).detach()            
            tok_utils.all_reduce(metrics_mse)
            metrics_mse /= cfg.world_size
            metrics_lpips = torch.stack([lpips]).detach()            
            tok_utils.all_reduce(metrics_lpips)
            metrics_lpips /= cfg.world_size
            metrics_vq = torch.stack([vq]).detach()            
            tok_utils.all_reduce(metrics_vq)
            metrics_vq /= cfg.world_size
            metrics_gan = torch.stack([gan]).detach()            
            tok_utils.all_reduce(metrics_gan)
            metrics_gan /= cfg.world_size
            
            avg_loss_mse.updata(metrics_mse[0].item())
            avg_loss_lpips.updata(metrics_lpips[0].item())
            avg_loss_vq.updata(metrics_vq[0].item())   
            avg_loss_gan.updata(metrics_gan[0].item()) 
            if total_step > cfg.disc_start :
                optimizer_idx = 1
            
            
        elif optimizer_idx == 1: 
            gan_loss, logits_real, logits_fake = criterion(
                videos,
                recons.detach(),
                x_aux, 
                extra_results_dict,   
                total_step,
                "discriminator"
            )
            loss = gan_loss  
            optimizer_idx = 0 
            # backward
            optimizer_d.zero_grad()
            scaler.scale(loss).backward() 
            scaler.step(optimizer_d)
            scaler.update() 
            
            metrics_gan = torch.stack([gan_loss]).detach()
            tok_utils.all_reduce(metrics_gan)
            metrics_gan /= cfg.world_size
            metrics_real = torch.stack([logits_real]).detach()
            tok_utils.all_reduce(metrics_real)
            metrics_real /= cfg.world_size
            metrics_fake = torch.stack([logits_fake]).detach()
            tok_utils.all_reduce(metrics_fake)
            metrics_fake /= cfg.world_size

            avg_loss_discr.updata(metrics_gan[0].item()) 
            avg_logit_real.updata(metrics_real[0].item()) 
            avg_logit_fake.updata(metrics_fake[0].item()) 
            
        # logging
        if cfg.rank == 0 and (
            total_step == (start_step+1)  or (total_step-1) % cfg.log_interval == 0
        ):
            codebook = torch.zeros(codebook_size,).cuda() 
            for i in range(len(codebook_idx)):
                codebook[codebook_idx[i]] += 1
            used_num = len(torch.nonzero(codebook))
            
            logging.info(
                f'Step: {total_step}/{cfg.num_steps} '
                f'mse: {avg_loss_mse.avg:.4f} '
                f'lpips: {avg_loss_lpips.avg:.4f} '
                f'gan: {avg_loss_gan.avg:.4f} '
                f'vq: {avg_loss_vq.avg:.4f} '
                f'discr: {avg_loss_discr.avg:.4} '
                f'real_logit: {avg_logit_real.avg:.4f} '
                f'fake_logit: {avg_logit_fake.avg:.4f} '
                f'used_rate: {used_num/codebook_size:.3f} '
                f'LR: {scheduler.get_lr()[0]} '
                f'Restart: {restart_time} '
            )

            avg_loss_mse.reset()
            avg_loss_lpips.reset()
            avg_loss_gan.reset()
            avg_loss_discr.reset()
            avg_loss_vq.reset() 
            avg_logit_real.reset()
            avg_logit_fake.reset()

        # visualization 
        if total_step == (start_step+1) or (total_step-1) % (cfg.log_interval) == 0:  
            os.makedirs(os.path.join('logs', cfg.log_dir,f'samples/rank_{cfg.rank}'), exist_ok=True)
            tok_utils.save_image(
                tensor=torch.stack([videos[-1:], recons[-1:]], dim=1).flatten(0, 1),
                path=os.path.join('logs', cfg.log_dir, f'samples/rank_{cfg.rank}/step_{total_step}.png'),
                nrow=8,
                normalize=(0,1),
            ) 
                
        # save last checkpoint
        if total_step == cfg.num_steps or total_step % (cfg.last_checkpoint_interval) == 0: 
            if cfg.rank == 0:
                os.makedirs(os.path.join('logs', cfg.log_dir, 'checkpoints'), exist_ok=True)
                save_inform = {'restart_time': restart_time,
                            'start_step':total_step+1,  
                            'log_dir': cfg.log_dir,
                            'model_state': model.module.state_dict(),
                            'criterion_state': criterion.module.state_dict(),
                            'scaler_state':scaler.state_dict()
                        }
                torch.save(save_inform, 'last.pth') 
                
        # save checkpoint
        if total_step == cfg.num_steps or total_step % (cfg.checkpoint_interval) == 0: 
            if cfg.rank == 0:
                os.makedirs(os.path.join('logs', cfg.log_dir, 'checkpoints'), exist_ok=True)
                save_inform = {'restart_time': restart_time,
                            'start_step':total_step+1,  
                            'log_dir': cfg.log_dir,
                            'model_state': model.module.state_dict(),
                            'criterion_state': criterion.module.state_dict(),
                            'scaler_state':scaler.state_dict()
                        }
                torch.save(save_inform, os.path.join('logs', cfg.log_dir, f'checkpoints/vae_step_{total_step}.pth')) 
                
                            
    logging.info('Congratulations! The training is completed!') 
    
    # barrier to ensure all ranks are completed
    torch.cuda.synchronize()
    dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vae')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--root', type=str, default=cfg.root)
    parser.add_argument('--stage', type=str, default=cfg.stage)
    parser.add_argument('--clustering_vq', action='store_true')
    parser.add_argument('--disc_start', type=int, default=cfg.disc_start)
    parser.add_argument('--perceptual_weight', type=float, default=cfg.perceptual_weight)
    parser.add_argument('--lpips_weight', type=float, default=cfg.lpips_weight)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    args = parser.parse_args()
    args_dict = vars(args)
    
    main(args_dict)
