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
from glob import glob

PANDASET_SEQ_LEN = 80
EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")
DYNAMIC_CLASSES = ALLOWED_RIGID_CLASSES + ALLOWED_NONRIGID_CLASSES

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_opts()

    dynamic_count = []
    for seq in sorted(glob(os.path.join(args.datapath, '*'))):
        pandaset = DataSet(args.datapath)
        seq = os.path.basename(seq)
        sequence = pandaset[seq]
        sequence.load()
    
        # ACTOR
        cuboids, poses = {}, []
        all_dynamics = {}
        for i in range(PANDASET_SEQ_LEN):
            curr_cuboids = sequence.cuboids[i]
            is_allowed_class = np.array([label in DYNAMIC_CLASSES for label in curr_cuboids["label"]])
            valid_mask = (~curr_cuboids["stationary"]) & is_allowed_class
            curr_cuboids = curr_cuboids[valid_mask]
            uids = np.array(curr_cuboids["uuid"]).tolist()
            labels = np.array(curr_cuboids["label"]).tolist()
            for uid, label in zip(uids, labels):
                all_dynamics[uid] = label
        
        rigid_cnt, non_rigid_cnt = 0, 0
        for uid, label in all_dynamics.items():
            if label in ALLOWED_RIGID_CLASSES:
                rigid_cnt += 1
            else:
                non_rigid_cnt += 1
        print((seq, rigid_cnt, non_rigid_cnt))
        dynamic_count.append((seq, rigid_cnt, non_rigid_cnt))
        
    dynamic_count = sorted(dynamic_count, key=lambda x: x[1]+x[2])
    with open(args.out, 'w') as f:
        for info in dynamic_count:
            f.writelines(f"{info[0]} rigids: {info[1]} nonrigids: {info[2]} \n")
    
