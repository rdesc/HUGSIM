import os
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread, imwrite
from glob import glob
from tqdm import tqdm
import cv2
import json
import argparse
from utils.fish2persp.get_virtual_perspective import virtual_perspect
from kitti360.annotation import Annotation3D
import math


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def loadCameraToPose(filename):
    # open file
    Tr = {}
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(line.strip().split())
            data = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
            data = np.concatenate((data,lastrow), axis=0)
            Tr[lineData[0]] = data
    return Tr['image_01:'], Tr['image_02:'], Tr['image_03:']

def load_data(datadir, sequence='2013_05_28_drive_0000_sync'):
    '''Load intrinstic matrix'''
    intrinstic_file = os.path.join(os.path.join(datadir, 'calibration'), 'perspective.txt')
    with open(intrinstic_file) as f:
        lines = f.readlines()
        for line in lines:
            lineData = line.strip().split()
            if lineData[0] == 'P_rect_00:':
                K_00 = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
                K_00 = K_00[:,:-1]
            elif lineData[0] == 'P_rect_01:':
                K_01 = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
                K_01 = K_01[:,:-1]
            elif lineData[0] == 'R_rect_01:':
                R_rect_01 = np.eye(4)
                R_rect_01[:3,:3] = np.array(lineData[1:]).reshape(3,3).astype(np.float64)

    '''Load extrinstic matrix'''
    CamPose_00 = {}
    CamPose_01 = {}
    CamPose_02 = {}
    CamPose_03 = {}
    extrinstic_file = os.path.join(datadir,os.path.join('data_poses',sequence))
    cam2world_file_00 = os.path.join(extrinstic_file,'cam0_to_world.txt')
    pose_file = os.path.join(extrinstic_file,'poses.txt')


    ''' Camera_00  to world coordinate '''
    with open(cam2world_file_00,'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(map(float,line.strip().split()))
            CamPose_00[int(lineData[0])] = np.array(lineData[1:]).reshape(4,4)
    
    ''' Camera_01 02 03 to world coordiante '''
    CamToPose_01, CamToPose_02, CamToPose_03 = loadCameraToPose(os.path.join(os.path.join(datadir, 'calibration'),'calib_cam_to_pose.txt'))
    poses = np.loadtxt(pose_file)
    frames = poses[:, 0]
    poses = np.reshape(poses[:, 1:], [-1, 3, 4])
    for frame, pose in zip(frames, poses):
        pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))
        pp = np.matmul(pose, CamToPose_01)
        CamPose_01[int(frame)] = np.matmul(pp, np.linalg.inv(R_rect_01))
        CamPose_02[int(frame)] = np.matmul(pose, CamToPose_02)
        CamPose_03[int(frame)] = np.matmul(pose, CamToPose_03)
    
    hom_K00, hom_K01 = np.eye(4), np.eye(4)
    hom_K00[:3, :3] = K_00
    hom_K01[:3, :3] = K_01

    return CamPose_00, CamPose_01, CamPose_02, CamPose_03, hom_K00, hom_K01, None, None

def get_kitti360_bbox(datadir, seq, start_index, end_index, inv_pose):
    annotation3D = Annotation3D(datadir, seq)
    rts = {}
    verts = {}
    cano_verts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5,  0.5], [0.5, -0.5, -0.5],
                            [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5]])
    for _, annotation in annotation3D.objects.items():
        for timestamp in annotation.keys():
            if timestamp < 0 or not (start_index <= timestamp < end_index):
                continue
            t = timestamp - start_index
            obj = annotation[timestamp]
            if not (obj.semanticId == 26 or obj.semanticId == 27 or obj.semanticId == 28):
                continue
            track_id = int(obj.instanceId)
            
            R = obj.R 
            bsize = np.linalg.norm(R, axis=0)
            vertices = cano_verts * bsize
            R = R / bsize
            T = obj.T

            P = np.eye(4)
            P[:3, :3] = R
            P[:3, 3] = T
            B2W = np.dot(inv_pose, P) # bbox --> mid frame Rt matrix

            if t not in rts:
                rts[t] = {}
            rts[t][track_id] = B2W.tolist()
            if track_id not in verts:
                verts[track_id] = vertices.tolist()

    return rts, verts

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--cams', nargs='+', type=int, required=True)
    return parser.parse_args()

def read_smt_package(out, cam, fn):
    smt_path = os.path.join(out, 'semantics', f'cam_{cam}_fisheye', fn)
    smt_package = {
        'smt': np.load(smt_path + '.npy'),
        'comp': imread(smt_path + '_comp.png'),
        'vis': imread(smt_path + '_vis.png'),
    }
    return smt_package
    
def save_smt_package(out, cam, fn, save_package):
    smt_path = os.path.join(out, 'semantics', f'cam_{cam}', fn)
    np.save(smt_path+'.npy', save_package['smt'])
    imwrite(smt_path+'_comp.png', save_package['comp'].astype(np.uint8))
    imwrite(smt_path+'_vis.png', save_package['vis'].astype(np.uint8))

def fish_eye(c2w, H, W, vk, out, cam, fn):
    img = cv2.imread(os.path.join(out, 'images', f'cam_{cam}_fisheye', f'{fn}.png'))
    smt_package = read_smt_package(out, cam, fn)
    if cam == '2':
        imgV, smtV_package, mask = virtual_perspect(H, W, vk, img, smt_package=smt_package, left=True)
    else:
        imgV, smtV_package, mask = virtual_perspect(H, W, vk, img, smt_package=smt_package, left=False)
    cv2.imwrite(os.path.join(out, 'images', f'cam_{cam}', f'{fn}.png'), imgV)
    save_smt_package(out, cam, fn, smtV_package)
    return c2w, vk, H, W


if __name__ == '__main__':

    args = get_opts()

    datadir = args.root
    output_dir = args.out
    start = args.start
    end = args.end
    
    p0, p1, p2, p3, k0, k1, _, _ = load_data(datadir, sequence='2013_05_28_drive_0000_sync')

    meta_data = {
        "camera_model": "OPENCV",
        "frames": [],
    }

    idx = 0

    frames = sorted(glob(os.path.join(output_dir, 'images', 'cam_0', '*.png')))
    frames_name = [os.path.basename(f).split('.')[0] for f in frames]
    # mid_frame = len(frames) // 2
    inv_pose = np.linalg.inv(p0[int(frames_name[0])])
    meta_data['inv_pose'] = inv_pose.tolist()
    meta_data['ref_frame'] = frames_name[0]

    rts, verts = get_kitti360_bbox(os.path.join(datadir, 'data_3d_bboxes'), '2013_05_28_drive_0000_sync', start, end, inv_pose)
    meta_data['verts'] = verts

    H, W = 360, 600
    fovx, fovy = 0.7 * np.pi, 0.6 * np.pi
    fx = fov2focal(fovx, W)
    fy = fov2focal(fovy, H)
    # fx, fy = 552, 552
    u0, v0 = W//2, H//2
    vk = np.array([[fx,     0,      u0,  0],
                   [0,      fy,     v0,  0],
                   [0,      0,      1,   0],
                   [0,      0,      0,   1]])

    available_cams = [f'cam_{cam}' for cam in args.cams]
    for fn in tqdm(frames_name):
        for cam in available_cams:
            if cam == 'cam_0':
                c2w = p0[int(fn)]
                k = k0
                h, w = 376, 1408
            elif cam == 'cam_1':
                c2w = p1[int(fn)]
                k = k1
                h, w = 376, 1408
            elif cam == 'cam_2':
                c2w = p2[int(fn)]
                c2w, k, h, w = fish_eye(c2w, H, W, vk, output_dir, '2', fn)
            elif cam == 'cam_3':
                c2w = p3[int(fn)]
                c2w, k, h, w = fish_eye(c2w, H, W, vk, output_dir, '3', fn)
            else:
                raise NotImplementedError
            c2w = np.dot(inv_pose, c2w)
            meta_data['frames'].append({
                "rgb_path": os.path.join('./images', cam, f'{fn}.png'),
                "camtoworld": c2w.tolist(),
                "intrinsics": k.tolist(),
                "width": w,
                "height": h,
                "timestamp": (int(fn) - start) * 0.1,
                "dynamics": rts.get(int(fn) - start, {})
            })
    
    with open(os.path.join(output_dir, 'meta_data.json'), 'w') as out_file:
        json.dump(meta_data, out_file, indent=4)