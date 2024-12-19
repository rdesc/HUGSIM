import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import cv2
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def load_params(visualizePCD=False):
    # left cam
    pcd_lt = np.load('utils/fish2persp/grid_fisheye_02.npy')
    u_orig_lt, v_orig_lt = np.meshgrid(np.arange(pcd_lt.shape[0]), np.arange(pcd_lt.shape[1]))
    u_orig_lt = u_orig_lt.flatten()
    v_orig_lt = v_orig_lt.flatten()
    pcd_lt = np.reshape(pcd_lt, (-1,4))

    mask_lt = np.load('utils/fish2persp/mask_left_fisheye.npy').astype(float)
    mask_lt = cv2.resize(mask_lt, (1400,1400), interpolation=cv2.INTER_NEAREST)
    mask_lt = (pcd_lt[:, 3] < 0.5) & (mask_lt.reshape(-1) < 0.5)

    pcd_lt = pcd_lt[mask_lt,:]
    u_orig_lt = u_orig_lt[mask_lt]
    v_orig_lt = v_orig_lt[mask_lt]

    # right cam
    pcd_rt = np.load('utils/fish2persp/grid_fisheye_03.npy')
    u_orig_rt, v_orig_rt = np.meshgrid(np.arange(pcd_rt.shape[0]), np.arange(pcd_rt.shape[1]))
    u_orig_rt = u_orig_rt.flatten()
    v_orig_rt = v_orig_rt.flatten()
    pcd_rt = np.reshape(pcd_rt, (-1,4))

    mask_rt = np.load('utils/fish2persp/mask_right_fisheye.npy').astype(float)
    mask_rt = cv2.resize(mask_rt, (1400,1400), interpolation=cv2.INTER_NEAREST)
    mask_rt = (pcd_rt[:, 3] < 0.5) & (mask_rt.reshape(-1) < 0.5)

    pcd_rt = pcd_rt[mask_rt,:]
    u_orig_rt = u_orig_rt[mask_rt]
    v_orig_rt = v_orig_rt[mask_rt]

    if visualizePCD:
        P = o3d.geometry.PointCloud()
        P.points = o3d.utility.Vector3dVector(pcd_lt[:,:3])
        o3d.visualization.draw_geometries([P])

    return pcd_lt, pcd_rt, u_orig_lt, v_orig_lt, u_orig_rt, v_orig_rt

pcd_lt, pcd_rt, u_orig_lt, v_orig_lt, u_orig_rt, v_orig_rt = load_params()


def virtual_perspect(H, W, K, img, smt_package=None, left=True):
    assert (img.shape[0] == img.shape[0]) and (img.shape[1] == img.shape[1])
    if left:
        pcd = pcd_lt
        u_orig, v_orig = u_orig_lt, v_orig_lt
    else:
        pcd = pcd_rt
        u_orig, v_orig = u_orig_rt, v_orig_rt
    # project the 3D points to the perspective camera
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    u = (K[0,0]*x)/z + K[0,2]
    v = (K[1,1]*y)/z + K[1,2]
    mask_u = np.logical_and(u>=0, u<W)
    mask_v = np.logical_and(v>=0, v<H)
    mask_uv = np.logical_and(mask_u, mask_v)
    u = u[mask_uv].astype(int)
    v = v[mask_uv].astype(int)
    u_orig = u_orig[mask_uv]
    v_orig = v_orig[mask_uv]

    imgV = np.ones((int(H),int(W),3)) * -1
    imgV[v,u] = img[v_orig, u_orig]
    mask = ~(np.sum(imgV == -1, axis=-1) == 0) # hole == True
    imgV[mask] *= 0
    imgV = imgV.astype(np.uint8)
    mask = mask.astype(np.uint8)
    imgV = cv2.inpaint(imgV, mask, 3, cv2.INPAINT_TELEA)

    if smt_package is not None:
        smt, comp, vis = smt_package['smt'], smt_package['comp'], smt_package['vis']
        smtV = np.zeros((int(H),int(W)), np.float32)
        smtV[v,u] = smt[v_orig, u_orig]
        smtV = cv2.inpaint(smtV, mask, 3, cv2.INPAINT_TELEA).clip(min=0, max=18).astype(int)
        
        compV = np.zeros((int(H),int(W),3), dtype=np.uint8)
        compV[v,u] = comp[v_orig, u_orig]
        compV = cv2.inpaint(compV, mask, 3, cv2.INPAINT_TELEA)
        
        visV = np.zeros((int(H),int(W),3), dtype=np.uint8)
        visV[v,u] = vis[v_orig, u_orig]
        visV = cv2.inpaint(visV, mask, 3, cv2.INPAINT_TELEA)
        smtV_package = {'smt': smtV, 'comp': compV, 'vis': visV}
    else:
        smtV_package = None
    return imgV, smtV_package, mask
    

if __name__=='__main__':
    
    H = W = 400
    fov = 0.6 * np.pi
    gamma = fov2focal(fov, H)

    u0, v0 = H//2, W//2

    K = np.array([[gamma, 0, u0],
                 [0, gamma, v0],
                 [0,0,1]])

    img = cv2.imread('/data1/datasets/KITTI-360/2013_05_28_drive_0000_sync/image_03/data_rgb/0000007788.png')
    imgV, _ = virtual_perspect(H, W, K, img, left=False)

    cv2.imwrite('./temp.png', imgV)
