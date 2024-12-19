import argparse
import os
from colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
import json
import numpy as np
import shutil

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    cameras_extrinsic_file = os.path.join(args.datapath, f"colmap_sparse_ba", "images.bin")
    cameras_intrinsic_file = os.path.join(args.datapath, f"colmap_sparse_ba", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    name2cam = {}
    for iid, image in cam_extrinsics.items():
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        c2w = np.linalg.inv(pose)
        name2cam[image.name] = {
            'extrinsic': c2w,
        }

    with open(os.path.join(args.datapath, "meta_data.json")) as f:
        meta_data = json.load(f)
    inv_pose = None
    for frame in meta_data['frames']:
        rgb_path = frame['rgb_path']
        rgb_name = rgb_path[9:]
        pose = name2cam[rgb_name]['extrinsic']
        if inv_pose is None:
            inv_pose = np.linalg.inv(pose)
            updated_inv_pose = inv_pose @ np.array(meta_data['inv_pose'])
            meta_data['inv_pose'] = updated_inv_pose.tolist()
        pose = inv_pose @ pose

        prev_pose = np.array(frame['camtoworld'])
        update_dynamics = {}
        for iid, prev_b2w in frame['dynamics'].items():
            prev_b2w = np.array(prev_b2w)
            b2c = np.linalg.inv(prev_pose) @ prev_b2w
            update_b2w = pose @ b2c
            update_dynamics[iid] = update_b2w.tolist()
            
        frame['camtoworld'] = pose.tolist()
        frame['dynamics'] = update_dynamics
    
    shutil.copy(os.path.join(args.datapath, "meta_data.json"), os.path.join(args.datapath, "meta_data_init.json"))
    with open(os.path.join(args.datapath, "meta_data.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)
