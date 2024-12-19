import os
import numpy as np
import yaml
from pandaset import DataSet
import cv2
import json
import mediapy as media
from tqdm import tqdm
import argparse
import pandas as pd
import open3d as o3d
from collections import defaultdict
from utils import (_pandaset_pose_to_matrix, _yaw_to_rotation_matrix, 
                   get_vertices, point_in_bbox, load_cam, cameras,
                   ALLOWED_RIGID_CLASSES, ALLOWED_NONRIGID_CLASSES)

PANDASET_SEQ_LEN = 80
EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")
DYNAMIC_CLASSES = ALLOWED_RIGID_CLASSES + ALLOWED_NONRIGID_CLASSES

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--video', action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    outdir = args.out
    render_video = args.video

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
    for cam in cameras:
        os.makedirs(os.path.join(outdir, 'images', cam), exist_ok=True)

    pandaset = DataSet(args.datapath)
    sequence = pandaset[args.seq]
    sequence.load()
    extrinsics = yaml.load(open(EXTRINSICS_FILE_PATH, "r"), Loader=yaml.FullLoader)
    inv_pose = np.linalg.inv(_pandaset_pose_to_matrix(sequence.camera['front_camera'].poses[0]))
    
    meta_data = {
        "camera_model": "OPENCV",
        "frames": [],
    }

    video_images = []
    
    start_timestamp = None

    # IMAGE
    for i in tqdm(range(PANDASET_SEQ_LEN)):
        cat_images = []
        for camera in cameras:
            im, im_name, height, width, intrinsic, pose, timestamp = \
                load_cam(sequence, camera, i, inv_pose, downsample=args.downsample)
            if start_timestamp is None:
                start_timestamp = timestamp
            cv2.imwrite(os.path.join(outdir, "images", camera, im_name), im)

            meta_data['frames'].append({
                "rgb_path": os.path.join("./images", camera, im_name),
                "camtoworld": pose.tolist(),
                "intrinsics": intrinsic.tolist(),
                "width": width,
                "height": height,
                "timestamp": timestamp - start_timestamp,
                "dynamics": {},
            })

            if render_video:
                im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                cat_images.append(cv2.resize(im2, (384,216)))
        
        if render_video:
            cat_images = cv2.vconcat([
                cv2.hconcat([cat_images[1], cat_images[0], cat_images[2]]),
                cv2.hconcat([cat_images[5], cat_images[3], cat_images[4]]),
            ])
            video_images.append(cat_images)

    # ACTOR
    cuboids, poses = {}, []
    for i in tqdm(range(PANDASET_SEQ_LEN)):
        curr_cuboids = sequence.cuboids[i]
        is_allowed_class = np.array([label in DYNAMIC_CLASSES for label in curr_cuboids["label"]])
        valid_mask = (~curr_cuboids["stationary"]) & is_allowed_class
        curr_cuboids = curr_cuboids[valid_mask]
        if not len(curr_cuboids):
            poses.append({})
            continue

        uuid = np.array(curr_cuboids["uuid"])
        label = np.array(curr_cuboids["label"])
        yaw = curr_cuboids["yaw"].astype(np.float32)
        rot = _yaw_to_rotation_matrix(yaw)
        pos_x = curr_cuboids["position.x"].astype(np.float32)  # x position of cuboid in world coords
        pos_y = curr_cuboids["position.y"].astype(np.float32)  # y position of cuboid in world coords
        pos_z = curr_cuboids["position.z"].astype(np.float32)  # z position of cuboid in world coords
        pos = np.vstack([pos_x, pos_y, pos_z]).T
        cuboid_poses = np.eye(4)[None].repeat(len(uuid), axis=0)
        cuboid_poses[:, :3, :3] = rot
        cuboid_poses[:, :3, 3] = pos
        cuboid_poses = inv_pose @ cuboid_poses
        width = curr_cuboids["dimensions.x"].astype(np.float32)  # width of cuboid in world coords
        length = curr_cuboids["dimensions.y"].astype(np.float32)  # length of cuboid in world coords
        height = curr_cuboids["dimensions.z"].astype(np.float32)  # height of cuboid in world coords
        dims = np.vstack([width, length, height]).T

        for cuboid_index in range(len(uuid)):
            cuboids[uuid[cuboid_index]] = {
                "label": label[cuboid_index],
                "dims": dims[cuboid_index],
                'verts': get_vertices(dims[cuboid_index])
            }
        curr_frame_poses = {}
        for cuboid_index in range(len(uuid)):
            curr_frame_poses[uuid[cuboid_index]] = cuboid_poses[cuboid_index]
        for uuid, pose in curr_frame_poses.items():
            for ii in range(6):
                meta_data['frames'][i*6 + ii]['dynamics'][uuid] = pose.tolist()
        poses.append(curr_frame_poses)
    meta_data['verts'] = {}
    for uuid, cuboid in cuboids.items():
        meta_data['verts'][uuid] = cuboid['verts'].tolist()

    print('Writing results...')
    # save meta data
    with open(os.path.join(outdir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)
    # save video
    if render_video:
        media.write_video(os.path.join(outdir, 'view.mp4'), video_images, fps=10)