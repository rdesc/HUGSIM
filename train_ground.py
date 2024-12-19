import os
import torch
from utils.loss_utils import l1_loss, ssim_loss
from gaussian_renderer import render_ground
import sys
from scene.ground_model import GroundModel 
import uuid
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from argparse import ArgumentParser
from torch.nn import CrossEntropyLoss
import json
import torchvision
from utils.dataset import HUGSIM_dataset, hugsim_collate, tocuda
from torch.utils.data import DataLoader
from scene import load_cameras
from scene.dataset_readers import fetchPly
from omegaconf import OmegaConf
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

# seedEverything()

results = {'train': {}, 'test': {}}

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# metrics
m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

def training(cfg):

    if cfg.semantic:
        semantic_ce = CrossEntropyLoss()
    
    train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)
    train_dataset = HUGSIM_dataset(train_cams, cfg.data_type)
    test_dataset = HUGSIM_dataset(test_cams, cfg.data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=hugsim_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=hugsim_collate)

    first_iter = 0
    prepare_output(cfg)
    pcd = fetchPly(os.path.join(cfg.source_path, 'ground_points3d.ply'))
    gaussians = GroundModel(cfg.model.sh_degree, pcd)

    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    progress_bar = tqdm(range(first_iter, cfg.train.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, cfg.train.iterations + 1):        

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        view_iid, prev_iid, gt_image, gt_semantic, gt_flow, gt_depth, mask = next(iter(train_dataloader))
        gt_image, gt_semantic, gt_flow, gt_depth, mask = gt_image.cuda(), tocuda(gt_semantic), tocuda(gt_flow), tocuda(gt_depth), tocuda(mask)
        viewpoint_cam = train_cams[view_iid]

        # Render
        render_pkg = render_ground(viewpoint_cam, gaussians, background)
        image, viewspace_point_tensor, info = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["info"]
        radii = info["radii"][0]
        visibility_filter = radii > 0
        viewspace_point_tensor.retain_grad()

        if iteration % 500 == 10:
            torchvision.utils.save_image(image, os.path.join(cfg.model_path, "ground", "save_train", f"{iteration}_{viewpoint_cam.image_name}.png"))

        # Loss
        loss = 0

        valid_mask = (gt_semantic <= 1)[0]
        image[:, ~valid_mask] *= 0
        gt_image[:, ~valid_mask] *= 0

        if cfg.semantic and gt_semantic is not None:
            semantic_map = render_pkg["feats"]
            semantic_loss = semantic_ce(semantic_map.permute(1,2,0)[valid_mask, :].view(-1, 20), gt_semantic[:, valid_mask].view(-1)) * 0.01
            loss += semantic_loss
        
        distort_3d_loss = 0 
        w2c = torch.linalg.inv(viewpoint_cam.c2w)
        points = gaussians.get_xyz
        c_points = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
        biases = -cfg.ground.min + cfg.ground.range * torch.rand(cfg.ground.n_sample, device='cuda')
        for bias in biases:
            mask = (bias < c_points[:, 2]) & (c_points[:, 2] < (bias + cfg.ground.grid_len)) 
            if torch.sum(mask) == 0:
                continue
            ys = c_points[mask, 1]
            distort_3d_loss += torch.std(ys)
        distort_3d_loss /= cfg.ground.n_sample
        loss += distort_3d_loss

        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - cfg.opt.lambda_dssim) * Ll1 + cfg.opt.lambda_dssim * ssim_loss(image, gt_image)
        loss += rgb_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                postfix = {"RGB": f"{rgb_loss:.{4}f}"}
                if cfg.semantic:
                    postfix["Semantic"] = f"{semantic_loss:.{4}f}"
                postfix["dist"] = f"{distort_3d_loss:.{4}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == cfg.train.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize()
            if (iteration in cfg.train.checkpoint_iterations):
                validation(iteration, cfg.model_path, gaussians, train_cams, test_cams, render_ground, background)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(cfg.model_path + '/ckpts', exist_ok=True)
                torch.save((gaussians.capture(), iteration), cfg.model_path + "/ckpts/ground_chkpnt" + str(iteration) + ".pth")
                gaussians.save_vis_ply(os.path.join(cfg.model_path, "point_cloud_vis/iteration_{}".format(iteration), "ground.ply"))

            # Optimizer step
            if iteration < cfg.train.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Densification
            if iteration < cfg.opt.densify_until_iter:
                grad = viewspace_point_tensor.grad[0].clone()
                grad[..., 0] *= info['width'] / 2.0
                grad[..., 1] *= info['height'] / 2.0
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_grad(grad, visibility_filter)

                if iteration > cfg.opt.densify_from_iter and iteration % cfg.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(cfg.opt.densify_grad_threshold, 0.005, 10, size_threshold)
                
                if iteration % cfg.opt.opacity_reset_interval == 0 or (cfg.model.white_background and iteration == cfg.opt.densify_from_iter):
                    gaussians.reset_opacity()


def prepare_output(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(os.path.join(args.model_path, "ground"), exist_ok = True)
    os.makedirs(os.path.join(args.model_path, "ground", "save_test"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "ground", "save_train"), exist_ok=True)
    OmegaConf.save(args, os.path.join(args.model_path, 'cfg.yaml'))

def validation(iteration, model_path, gaussians, train_cameras, test_cameras, renderFunc, background):

    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : test_cameras}, 
                            {'name': 'train', 'cameras' : train_cameras})

    for config in validation_configs:
        # if config['cameras'] and len(config['cameras']) > 0:
        l1_test = 0
        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        for viewpoint in config['cameras']:
            gt_image = viewpoint.original_image.cuda()
            image = torch.clamp(renderFunc(viewpoint, gaussians, background)["render"], 0.0, 1.0)
            mask = viewpoint.semantic2d.cuda() > 1
            image[:, mask[0]] *= 0
            gt_image[:, mask[0]] *= 0
            l1_test += l1_loss(image, gt_image).mean()
            image = image[None, ...]
            gt_image = gt_image[None, ...]
            psnr_test += m_psnr(image, gt_image)
            ssim_test += m_ssim(image, gt_image)
            lpips_test += m_lpips(image, gt_image)

            if config['name'] == 'test':
                torchvision.utils.save_image(image, os.path.join(model_path, "ground", "save_test", f"{viewpoint.image_name}.png"))
        
        psnr_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])          
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {format(l1_test, '.4f')} "
                f"PSNR {format(psnr_test, '.4f')} SSIM {format(ssim_test, '.4f')} Lpips {format(lpips_test, '.4f')}")
        
        results[config['name']][iteration] = {
            'psnr': psnr_test.item(),
            'ssim': ssim_test.item(),
            'lpips': lpips_test.item(),
            'l1': l1_test.item()
        }

    torch.cuda.empty_cache()

    with open(os.path.join(model_path, 'ground', 'results.json'), 'w') as wf:
        json.dump(results, wf, indent=4)
        
def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--data_cfg", type=str, default="./configs/nusc.yaml")
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.base_cfg), OmegaConf.load(args.data_cfg))
    if len(args.source_path) > 0:
        cfg.source_path = args.source_path
    if len(args.model_path) > 0:
        cfg.model_path = args.model_path
        
    print("Optimizing " + args.model_path)
    training(cfg)
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
    
