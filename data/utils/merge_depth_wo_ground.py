import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--total', type=int, default=1500000)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                        unproject pixels                             #
    ##########################################################################
    
    points, colors = [], []
    sample_per_frame = args.total // len(meta_data["frames"])
    for frame in tqdm(meta_data["frames"]):
        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        rgb_path = frame["rgb_path"]
        frame_cam = frame["rgb_path"].split("/")[-2]
        im = np.array(imread(os.path.join(args.out, rgb_path)))
        depth_path = os.path.join(
            args.out,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()

        x = np.arange(0, depth.shape[1])  # generate pixel coordinates
        y = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

        # unproject depth to pointcloud
        x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
        y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
        z = depth.reshape(-1)
        local_points = np.stack([x, y, z], axis=1)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # mask dynamic
        mask_path = os.path.join(args.out,
                                rgb_path.replace('images', 'masks').replace('./', '').replace('.jpg', '.npy').replace('.png', '.npy'))
        if os.path.exists(mask_path):
            dynamic_mask = np.load(mask_path).reshape(-1)

        # non-ground semantics
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            smt_mask = smts > 1
            
        # mask = dynamic_mask & smt_mask
        mask = smt_mask
        local_points = local_points[mask]
        local_colors = local_colors[mask]

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            continue
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(args.out, "points3d.ply"), pcd)
