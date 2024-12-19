import os
import numpy as np
import cv2
import json
import mediapy as media
from tqdm import tqdm
import argparse
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import torch
import shutil
from scipy.spatial.transform import Rotation as R
from nusc.utils import (AVAILABLE_CAMERAS, WLH_TO_LWH, ALLOWED_CLASSES,
                   _rotation_translation_to_pose, find_all_sample_data, find_all_sample, 
                   get_vertices, point_in_bbox, frame_check, get_sample_pose, load_cam,
                   traj_dict_to_list, get_box)


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--video', action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()
    outdir = args.out

    meta_data = {
        "camera_model": "OPENCV",
        'verts': {},
        "frames": [],
    }

    nusc = NuScenes(version=args.version, dataroot=args.datapath, verbose=True)
    scene = nusc.get("scene", nusc.field2token("scene", "name", args.seq)[0])
    first_sample = nusc.get("sample", scene["first_sample_token"])
    
    ##########################################################################
    #                        Pre-preparations                         #
    #          Read all sensors relative pose in first frame          #
    #   Lidar is only used for extrating camera height relative to ground    #
    ##########################################################################
    
    # Lidar
    lidar_data = nusc.get('sample_data', first_sample["data"]["LIDAR_TOP"])
    calibrated_lidar_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    ego_pose_data = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
    lidar_pose = _rotation_translation_to_pose(
        calibrated_lidar_data["rotation"], calibrated_lidar_data["translation"]
    ) # lidar2ego
    lidar2w = ego_pose @ lidar_pose # lidar2world
    
    # Ground
    lidar_fn = os.path.join(args.datapath, lidar_data["filename"])
    points = np.fromfile(lidar_fn, dtype=np.float32).reshape([-1, 5])[:, :3]
    ego_mask = (np.abs(points[:, 0]) < 1.5) & (np.abs(points[:, 1]) < 2.5)
    ground_mask = (np.abs(points[:, 0]) < 3) & (np.abs(points[:, 1]) < 6)
    points = points[ground_mask & (~ego_mask)]
    # points = (lidar2w[:3, :3] @ points.T).T + lidar2w[:3, 3]
    points = (lidar_pose[:3, :3] @ points.T).T + lidar_pose[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    a, b, c, d = plane_model
    
    # save first frame lidar pcd
    o3d.io.write_point_cloud(os.path.join(outdir, 'ground_lidar.ply'), pcd)
    
    # Camera
    all_campose = {}
    for cam in AVAILABLE_CAMERAS:
        cam_data = nusc.get('sample_data', first_sample["data"][cam])
        calibrated_cam_data = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        cam_pose = _rotation_translation_to_pose(
            calibrated_cam_data["rotation"], calibrated_cam_data["translation"]
        )
        all_campose[cam] = cam_pose
    front_cam_t = all_campose['CAM_FRONT'][:3, 3]
    height = -(a*front_cam_t[0] + b*front_cam_t[1] + d) / c
    
    n = np.array([a,b,c])
    front_cam_z = all_campose['CAM_FRONT'][:3, 0]
    cos_theta = np.dot(n, front_cam_z) / (np.linalg.norm(n) * np.linalg.norm(front_cam_z))
    pitch_angle = np.arccos(cos_theta)
    rect_pitch = np.pi / 2 - pitch_angle
    rect_mat = R.from_euler('x', rect_pitch).as_matrix() # rectify under camera coord
    
    front_cam_info = {
        "height": front_cam_t[2] - height, 
        "rect_mat": rect_mat.tolist(),
    }
    with open(os.path.join(outdir, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f)
    
    # save camera relative pose for rigid bundle adjustment
    cam_rigid = dict()
    cam_rigid["ref_camera_id"] = 1
    rigid_cam_list = []
    ref_extrinsic = all_campose['CAM_FRONT']
    for iid, cam_name in enumerate(AVAILABLE_CAMERAS):
        rigid_cam = dict()
        rigid_cam["camera_id"] = iid+1

        cur_extrinsic = all_campose[cam_name]
        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic
        r = R.from_matrix(rel_extrinsic[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = f'{cam_name}/'        
        rigid_cam['cam_from_rig_rotation'] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam['cam_from_rig_translation'] = [rel_extrinsic[0, 3], rel_extrinsic[1, 3], rel_extrinsic[2, 3]]
        rigid_cam_list.append(rigid_cam)

    cam_rigid["cameras"] = rigid_cam_list
    with open(os.path.join(outdir, "cam_rigid_config.json"), "w") as f:
        json.dump([cam_rigid], f, indent=4)   
    
    
    ##########################################################################
    #                    Read all frames in sequences                       #
    ##########################################################################
    
    samples = find_all_sample(nusc, first_sample)[args.start:args.end]
    fff_sample_data = nusc.get("sample_data", samples[0]['data']["CAM_FRONT"])
    inv_pose = np.linalg.inv(get_sample_pose(nusc, fff_sample_data)[0])
    meta_data['inv_pose'] = inv_pose.tolist()
    
    shutil.rmtree(os.path.join(outdir, 'images'), ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
    for cam in AVAILABLE_CAMERAS:
        os.makedirs(os.path.join(outdir, 'images', cam), exist_ok=True)

    tracks = defaultdict(list)
    for sample in samples:
        for box_token in sample['anns']:
            instance_token, pose, _ = get_box(nusc, box_token, inv_pose)
            tracks[instance_token].append(pose)
    dynamic_instance = set()
    for instance_token, traj_list in tracks.items():
        poses = torch.from_numpy(np.stack(traj_list).astype(np.float32))
        dynamic = np.max(traj_list[0][:3, 3] - traj_list[-1][:3, 3]) > 2
        if dynamic:
            dynamic_instance.add(instance_token)

    video_images = []
    start_time = -1
    for i, sample in tqdm(enumerate(samples)):

        dynamics = {}
        for box_token in sample["anns"]:
            instance_token, pose, lhw = get_box(nusc, box_token, inv_pose)
            if instance_token not in dynamic_instance:
                continue
            dynamics[instance_token] = pose.tolist()
            if instance_token not in meta_data['verts']:
                meta_data['verts'][instance_token] = get_vertices(lhw).tolist()

        cat_images = []
        for cam in AVAILABLE_CAMERAS:
            sample_data = nusc.get('sample_data', sample['data'][cam])
            im, im_name, height, width, intrinsic, pose = \
                load_cam(nusc, sample_data, inv_pose, args.datapath, downsample=args.downsample)
            im_name = str(i).zfill(5) + '.jpg'
            cv2.imwrite(os.path.join(outdir, "images", cam, im_name), im)

            timestamp = sample["timestamp"] / 1e6
            if start_time < 0:
                start_time = timestamp
            timestamp -= start_time
            meta_data['frames'].append({
                "rgb_path": os.path.join("./images", cam, im_name),
                "camtoworld": pose.tolist(),
                "intrinsics": intrinsic.tolist(),
                "width": width,
                "height": height,
                'timestamp': timestamp,
                "dynamics": dynamics,
            })

            if args.video:
                im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im2 = cv2.resize(im2, (400, 225))
                cat_images.append(im2)

        if args.video:
            cat_images = cv2.vconcat([
                cv2.hconcat([cat_images[1], cat_images[0], cat_images[2]]),
                cv2.hconcat([cat_images[5], cat_images[3], cat_images[4]]),
            ])
            video_images.append(cat_images)

    print('Writing results...')
    # save meta data
    with open(os.path.join(outdir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)
    if args.video:
        media.write_video(os.path.join(outdir, 'view.mp4'), video_images, fps=12)
            