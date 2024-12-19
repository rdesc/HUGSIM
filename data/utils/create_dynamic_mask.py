import numpy as np
import json
import os
from imageio.v2 import imread, imwrite
import argparse
import cv2

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    return parser.parse_args()

def checkcorner(corner, h, w):
    if np.all(corner < 0) or (corner[0] >= h and corner[1] >= w):
        return False
    else:
        return True

def main():
    args = get_opts()
    basedir = args.data_path
    os.makedirs(os.path.join(basedir, 'masks'), exist_ok=True)
    if args.data_type == 'kitti360':
        cameras = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
    elif args.data_type == 'pandaset':
        AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")
        cameras = [cam + "_camera" for cam in AVAILABLE_CAMERAS]
    elif args.data_type == 'nuscenes':
        cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    elif args.data_type == 'waymo':
        cameras = ['cam_1', 'cam_2', 'cam_3']
    else:
        raise NotImplementedError
    for cam in cameras:
        os.makedirs(os.path.join(basedir, 'masks', cam), exist_ok=True)
    # Opening JSON file
    with open(os.path.join(basedir, "meta_data.json")) as f:
        meta_data = json.load(f)

    verts = meta_data['verts']
    for f in meta_data['frames']:
        rgb_path = f['rgb_path']
        c2w = np.array(f['camtoworld'])
        intr = np.array(f['intrinsics'])
        w2c = np.linalg.inv(c2w)

        smt = np.load(os.path.join(basedir, rgb_path.replace('images', 'semantics').replace('.jpg', '.npy')).replace('.png', '.npy'))
        car_mask = (smt == 11) | (smt == 12) | (smt == 13) | (smt == 14) | (smt == 15) | (smt == 18)
        mask = np.zeros_like(car_mask).astype(np.bool_)

        for iid, rt in f['dynamics'].items():
            H, W = mask.shape[0], mask.shape[1]
            rt = np.array(rt)
            points = np.array(verts[iid])
            points = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            xyz_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            valid_depth = xyz_cam[:, 2] > 0
            xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
            xy_screen  = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]
            valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
            valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
            valid_pixel = valid_x & valid_y & valid_depth

            if valid_pixel.any():
                xy_screen = np.round(xy_screen).astype(int)
                bbox_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 1, 4, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[2, 3, 6, 7, 2]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 7, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[1, 3, 6, 4, 1]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 3, 1, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[5, 4, 6, 7, 5]]], 1)
                bbox_mask = bbox_mask & car_mask
                mask = mask | (bbox_mask != 0)

        save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
        np.save(save_path.replace('.jpg', '.npy').replace('.png', '.npy'), ~mask)
        imwrite(save_path+'.png', (~mask).astype(np.uint8) * 255)

if __name__ == "__main__":
    main()