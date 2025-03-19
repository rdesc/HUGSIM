# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from apex import amp
from runx.logx import logx
import numpy as np
import torch
import argparse
import os
import sys
import time
import fire
from utils.config import assert_and_infer_cfg, cfg
from utils.misc import AverageMeter, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch
from utils.progress_bar import printProgressBar
from models.loss.utils import get_loss
from models.model_loader import load_model
from library.datasets.get_dataloaders import return_dataloader
import models
import warnings
from PIL import Image
import torchvision.transforms as standard_transforms
from torchvision.utils import save_image

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
torch.backends.cudnn.benchmark = True    
palette = [128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            70, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
color_mapping = palette
    
def set_apex_params(local_rank):
    """
    Setting distributed parameters for Apex
    """
    world_size = int(os.environ['WORLD_SIZE'])  # worker RANK and WORLD_SIZE are assigned automatically with torchrun
    global_rank = int(os.environ['RANK'])

    print('GPU {} has Rank {}'.format(local_rank, global_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    return world_size, global_rank

def colorize(image_array):
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_mapping)
    return new_mask

def inference(val_loader, net, arch, loss_fn, output_dir, epoch=0, calc_metrics=False):
    """
    Inference over dataloader on network
    """

    len_dataset = len(val_loader)
    net.eval()
    val_loss = AverageMeter()

    for val_idx, data in enumerate(val_loader):
        input_images, labels, edge, img_names, _, raw_images = data 
        
        # Run network
        assets, _ = \
            eval_minibatch(data, net, loss_fn, val_loss, calc_metrics, val_idx)
        
        for i in range(len(assets['predictions'])):
            prediction = assets['predictions'][i]
            prediction_fn = img_names[i] + '.npy'
            prediction_fn_rgb = img_names[i] + '_vis.png'
            composited_fn = img_names[i] + '_comp.png'

            np.save(os.path.join(output_dir, prediction_fn), prediction)
            prediction_pil = colorize(prediction)
            prediction_pil.save(os.path.join(output_dir, prediction_fn_rgb))
            input_image = raw_images[i].detach().cpu()
            input_image = standard_transforms.ToPILImage()(input_image)
            composited = Image.blend(input_image, prediction_pil.convert("RGB"), 0.4)
            composited.save(os.path.join(output_dir, composited_fn))
        
            if val_idx+1 < len_dataset:
                printProgressBar(val_idx + 1, len_dataset, 'Progress')


def main(input_dir, output_dir, model_path, has_edge=False, model_summary=False, arch='ocrnet.AuxHRNet', 
         hrnet_base=18, num_workers=4, split='val', batch_size=2, crop_size='1024,2048', 
         apex=True, syncbn=True, fp16=True, local_rank=0):

    #Distributed processing
    if apex:
        world_size, global_rank = set_apex_params(local_rank)
    else:
        world_size = 1
        global_rank = 0  
        local_rank = 0  
        
    #Logging
    logx.initialize(logdir=output_dir,
                    tensorboard=True,
                    global_rank=global_rank)

    #Build config
    assert_and_infer_cfg(output_dir, global_rank, apex, syncbn, arch, hrnet_base,
                         fp16, has_edge)
    
    #Dataloader
    print(input_dir)
    val_loader = return_dataloader(num_workers, 1, input_dir)
    print(len(val_loader))

    #Loss function
    loss_fn = get_loss(has_edge)

    assert model_path is not None, 'need pytorch model for inference'
    
    #Load Network
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    logx.msg("Loading weights from: {}".format(model_path))
    net = models.get_net(arch, loss_fn)
    if fp16:
        net = amp.initialize(net, opt_level='O1', verbosity=0)
    net = models.wrap_network_in_dataparallel(net, apex)
    #restore_net(net, checkpoint, arch)
    load_model(net, checkpoint)
    
    torch.cuda.empty_cache()
    
    #Run inference
    inference(val_loader, net, arch, loss_fn, output_dir, epoch=0)


if __name__ == '__main__':
    fire.Fire(main)
