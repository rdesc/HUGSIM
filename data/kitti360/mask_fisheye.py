from utils.fish2persp.get_virtual_perspective import load_params
import argparse
from glob import glob
from imageio.v2 import imread, imwrite
import numpy as np
import cv2
from tqdm import tqdm
import os

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    
    pcd_lt = np.load('utils/fish2persp/grid_fisheye_02.npy')
    u_orig_lt, v_orig_lt = np.meshgrid(np.arange(pcd_lt.shape[0]), np.arange(pcd_lt.shape[1]))
    u_orig_lt = u_orig_lt.flatten()
    v_orig_lt = v_orig_lt.flatten()
    pcd_lt = np.reshape(pcd_lt, (-1,4))
    mask_lt = np.load('utils/fish2persp/mask_left_fisheye.npy').astype(float)
    mask_lt = cv2.resize(mask_lt, (1400,1400), interpolation=cv2.INTER_NEAREST)
    mask_lt = (pcd_lt[:, 3] < 0.5) & (mask_lt.reshape(-1) < 0.5)
    mask_lt = mask_lt.reshape((1400, 1400))
    
    pcd_rt = np.load('utils/fish2persp/grid_fisheye_03.npy')
    u_orig_rt, v_orig_rt = np.meshgrid(np.arange(pcd_rt.shape[0]), np.arange(pcd_rt.shape[1]))
    u_orig_rt = u_orig_rt.flatten()
    v_orig_rt = v_orig_rt.flatten()
    pcd_rt = np.reshape(pcd_rt, (-1,4))
    mask_rt = np.load('utils/fish2persp/mask_right_fisheye.npy').astype(float)
    mask_rt = cv2.resize(mask_rt, (1400,1400), interpolation=cv2.INTER_NEAREST)
    mask_rt = (pcd_rt[:, 3] < 0.5) & (mask_rt.reshape(-1) < 0.5)
    mask_rt = mask_rt.reshape((1400, 1400))
    
    for cam, mask in [('cam_2_fisheye', mask_lt), ('cam_3_fisheye', mask_rt)]:
        for fn in tqdm(glob(os.path.join(args.out, 'images', cam, "*.png"))):
            im = imread(fn)
            im[~mask] *= 0
            imwrite(fn, im)
            
            