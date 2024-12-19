import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pickle


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--total", type=int, default=1500000)
    parser.add_argument("--datatype", type=str, default="nuscenes")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.out, "meta_data.json"), "r") as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                        unproject pixels                             #
    ##########################################################################
    
    points, colors = [], []
    sample_per_frame = args.total // len(meta_data["frames"])
    front_cam_poses = []
    for frame in tqdm(meta_data["frames"]):
        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        if args.datatype == "nuscenes":
            if "/CAM_FRONT/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "pandaset":
            if "/front_camera/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "waymo":
            if "/cam_1/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "kitti360":
            if "/cam_0/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        else:
            raise NotImplementedError

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

        # ground semantics
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            mask = smts <= 1
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

    ##########################################################################
    #                    Multi-Plane Ground Model                       #
    ##########################################################################
    
    # Read front cam poses
    if args.datatype == "kitti360":
        front_cam_height = 1.55
    elif args.datatype == 'pandaset':
        front_cam_height = 2.2
    else:
        with open(os.path.join(args.out, "front_info.json"), "r") as f:
            front_info = json.load(f)
        front_cam_height = front_info["height"]
        front_rect_mat = front_info["rect_mat"]
    front_cam_poses = np.stack(front_cam_poses)
    # front_cam_poses[:, :3, :3] = np.einsum('ij, njk -> nik', front_rect_mat, front_cam_poses[:, :3, :3])
    
    # Init ground point cloud
    points_cam_dist = np.sqrt(
        np.sum(
            (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :])
            ** 2,
            axis=-1,
        )
    )
    
    # nearest cam
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
    nearest_c2w = front_cam_poses[nearest_cam_idx] # (N, 4, 4)
    nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx] # (N, 4, 4)
    points_local = (
        np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points)
        + nearest_w2c[:, :3, 3]
    ) # (N, 3)
    points_local[:, 1] = front_cam_height
    points = (
        np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
        + nearest_c2w[:, :3, 3]
    ) # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(args.out, "ground_points3d.ply"), pcd)
    
    # Get high level command
    forecast = 20
    threshold = 2.5
    high_level_commands = []
    for i, cam_pose in enumerate(front_cam_poses):
        if i + forecast < front_cam_poses.shape[0]:
            forecast_campose = front_cam_poses[i + forecast]
        else:
            forecast_campose = front_cam_poses[-1]
        inv_cam_pose = np.linalg.inv(cam_pose)
        forecast_in_curr = inv_cam_pose @ forecast_campose
        if forecast_in_curr[0, 3] > threshold:
            high_level_commands.append(0) # right
        elif forecast_in_curr[0, 3] < -threshold:
            high_level_commands.append(1) # left
        else:
            high_level_commands.append(2) # forward

    print(high_level_commands)
    with open(os.path.join(args.out, "ground_param.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)
