""" Adapted from:
    https://github.com/LTH14/mar/blob/main/main_cache.py
    https://github.com/bytedance/1d-tokenizer/blob/main/scripts/pretokenization.py
"""
import argparse
import datetime
import numpy as np
from PIL import Image
import torch.distributed as dist

import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import utils.misc as misc
from tqdm import tqdm
import json
import glob

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VQ codes', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='cache_alitok', help='path to cached latents')
    parser.add_argument("--ten_crop", action='store_true', help="whether using random crop")
    parser.add_argument("--flip_flag", action='store_true', help="whether using flip")

    return parser


def convert_json_to_jsonl(input_pattern, output_file):
    with open(output_file, 'w') as outfile:
        for filename in tqdm(glob.glob(input_pattern)):
            with open(filename, 'r') as infile:
                data = json.load(infile)
                for item in data:
                    json.dump(item, outfile)
                    outfile.write('\n')


@torch.no_grad()
def main(args):
    os.makedirs(args.cached_path, exist_ok=True)
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.ten_crop:
        crop_size = int(args.img_size * 1.1)
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.img_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops])),
        ])
    else:
        # augmentation following DiT and ADM
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # MaskGIT-VQ expects input in range of [0, 1]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )
    
    # AliTok
    from modeling.alitok import AliTok
    tokenizer = AliTok()
    checkpoint = torch.load('weights/AliTok.pth', map_location='cpu')  
    tokenizer.load_state_dict(checkpoint, strict=True) 
    del checkpoint  
    
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)
 
    print(f"Start caching latents, {args.rank}, {args.gpu}")
    start_time = time.time() 
    for samples, target, paths in tqdm(data_loader_train):
        samples = samples.to(device, non_blocking=True)
        
        if args.ten_crop:
            samples_all = samples.flatten(0, 1) 
            target_all = target.unsqueeze(1).repeat(1, 10).flatten(0, 1)  
            
            # flip 
            if args.flip_flag:
                samples_all_flip = samples_all.flip(dims=[-1])
        else:
            samples_all = torch.cat([samples, torch.flip(samples, dims=[-1])])
            target_all = torch.cat([target, target])
            
        with torch.no_grad(): 
            z_quantized, result_dict  = tokenizer.encode(samples_all, tokenizer.latent_tokens)
            codes = result_dict["min_encoding_indices"].reshape(target_all.shape[0], -1) 
            if args.flip_flag:
                z_quantized_flip, result_dict_flip  = tokenizer.encode(samples_all_flip, tokenizer.latent_tokens) 
                codes_flip = result_dict["min_encoding_indices"].reshape(target_all.shape[0], -1)  
        if args.flip_flag:
            codes = torch.stack([codes, codes_flip], 1) 
            if args.ten_crop:
                codes = codes.reshape(len(paths), 10*2, -1)  
            else:
                codes = codes.reshape(len(paths), 2, -1)   
        else:
            if args.ten_crop:
                codes = codes.reshape(len(paths), 10, -1)  
            else: 
                codes = codes.reshape(len(paths), 1, -1) 
        
        for i, path in enumerate(paths):  
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True) 

            np.savez(save_path, codes=codes[i].cpu().numpy()) 
        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()
            
    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
