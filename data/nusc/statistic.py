from nuscenes.nuscenes import NuScenes
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils import get_box
import argparse

camera_list = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
]

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

args = get_opts()
dataroot = args.datapath
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

idx = []
dynamic_count = []
for i in tqdm(range(len(nusc.scene))):
    scene = nusc.scene[i]
    s = scene['name']
    desc = scene['description'].lower()
    if ('wait' in desc) or ('night' in desc) or ('rain' in desc):
        continue

    token = scene['first_sample_token']
    tracks = defaultdict(list)
    while token != '':
        sample = nusc.get('sample', token)
        cat_images = []

        for ann_token in sample['anns']:
            instance_token, pose, _ = get_box(nusc, ann_token, np.eye(4))
            tracks[instance_token].append(pose)
        
        token = sample['next']
            
    dynamic_instance = set()
    for instance_token, traj_list in tracks.items():
        dynamic = np.max(traj_list[0][:3, 3] - traj_list[-1][:3, 3]) > 2
        if dynamic:
            dynamic_instance.add(instance_token)
    
    dynamic_count.append((s, len(dynamic_instance)))
    
dynamic_count = sorted(dynamic_count, key=lambda x: x[1])
with open(args.out, 'w') as f:
    for info in dynamic_count:
        f.writelines(f"scene name: {info[0]} dynamics: {info[1]} \n")
