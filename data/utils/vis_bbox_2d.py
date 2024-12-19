import numpy as np
import open3d as o3d
import json
from imageio.v2 import imwrite
import matplotlib.pyplot as plt 
import os
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

def main():
    args = get_opts()
    dataset = args.out
    meta_data_pth = os.path.join(dataset, 'meta_data.json')

    with open(meta_data_pth, 'r') as f:
        meta_data = json.load(f)
    frames = meta_data['frames']
    verts = meta_data['verts']

    for frame in tqdm(frames):
        rgb_path = frame['rgb_path']
        c2w = np.array(frame['camtoworld'])
        intr = np.array(frame['intrinsics'])
        w2c = np.linalg.inv(c2w)
        im = cv2.imread(os.path.join(dataset, rgb_path))

        for objIds, rt in frame['dynamics'].items():
            # Create Line Set
            rt = np.array(rt)
            points = np.array(verts[objIds])
            points = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            connections = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                        [1, 4], [0, 5], [3, 6], [2, 7]]
            xyz_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
            if np.any(xyz_screen[:, 2] < 0):
                continue
            xy_screen  = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]
            xy_screen = xy_screen.astype(int)
            for connection in connections:
                line = xy_screen[connection, :].tolist()
                cv2.line(im, line[0], line[1], color=(0,0,255), thickness=1)
            cv2.putText(im, objIds[:5], (xy_screen[0] + xy_screen[5]) // 2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        save_name = '_'.join(rgb_path.split('/')[-2:])
        visualize = os.path.join(dataset, 'vis_bbox')
        os.makedirs(visualize, exist_ok=True)
        cv2.imwrite(os.path.join(visualize, save_name), im)

if __name__ == '__main__':
    main()