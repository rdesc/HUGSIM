import os
import numpy as np
import re
import yaml
import sys

import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

def checkfile(filename):
    if not os.path.isfile(filename):
        raise RuntimeError('%s does not exist!' % filename)

def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def loadCalibrationCameraToPose(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');

    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))

    # close file
    fid.close()
    return Tr


def readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

class Camera:
    def __init__(self):
        
        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            # consider the rectification for perspective cameras
            if self.cam_id==0 or self.cam_id==1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id==2 or self.cam_id==3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')


    def world2cam(self, points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        if ndim==2:
            points = points[0]

        return points

    def cam2image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError
    
    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u,v,depth = self.cam2image(points_local)

        return (u,v), depth 

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 
        obj3d.generateMeshes()


class CameraPerspective(Camera):

    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=0):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==0 or cam_id==1)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraPerspective, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load perspective intrinsics '''
    
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % self.cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % self.cam_id:
                R_rect = np.eye(4) 
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_%02d:" % self.cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)
    
        self.K = K
        self.width, self.height = width, height
        self.R_rect = R_rect

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth

class CameraFisheye(Camera):
    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=2):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==2 or cam_id==3)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'image_%02d.yaml' % cam_id)
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraFisheye, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load fisheye intrinsics '''

        intrinsics = readYAMLFile(intrinsic_file)

        self.width, self.height = intrinsics['image_width'], intrinsics['image_height']
        self.fi = intrinsics

    def cam2image(self, points):
        ''' camera coordinate to image plane '''
        points = points.permute(1, 0)
        norm = torch.norm(points, dim=1, p=2)

        x = points[:,0] / norm
        y = points[:,1] / norm
        z = points[:,2] / norm

        x = x / (z+self.fi['mirror_parameters']['xi'])
        y = y / (z+self.fi['mirror_parameters']['xi'])

        k1 = self.fi['distortion_parameters']['k1']
        k2 = self.fi['distortion_parameters']['k2']
        gamma1 = self.fi['projection_parameters']['gamma1']
        gamma2 = self.fi['projection_parameters']['gamma2']
        u0 = self.fi['projection_parameters']['u0']
        v0 = self.fi['projection_parameters']['v0']

        ro2 = x*x + y*y
        x = x * (1 + k1*ro2 + k2*ro2*ro2)
        y = y * (1 + k1*ro2 + k2*ro2*ro2)

        x = gamma1*x + u0
        y = gamma2*y + v0

        return x, y, norm * points[:,2] / torch.abs(points[:,2])

if __name__=="__main__":
    import cv2
    import matplotlib.pyplot as plt

    kitti360Path = '/Users/yliao/data/KITTI-360/'
    seq = 0
    cam_id = 3
    sequence = '2013_05_28_drive_%04d_sync'%seq
    # perspective
    if cam_id == 0 or cam_id == 1:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye
    elif cam_id == 2 or cam_id == 3:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Invalid Camera ID!')


    H = 1400
    W = 1400
    # [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # [H*W]
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5

    # [3, H*W]
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)

    grid = torch.from_numpy(pixels)

    map_dist = []
    z_dist = []
    
    k1 = camera.fi['distortion_parameters']['k1']
    k2 = camera.fi['distortion_parameters']['k2']
    gamma1 = camera.fi['projection_parameters']['gamma1']
    gamma2 = camera.fi['projection_parameters']['gamma2']
    u0 = camera.fi['projection_parameters']['u0']
    v0 = camera.fi['projection_parameters']['v0']
    mirror = camera.fi['mirror_parameters']['xi']

    for ro2 in torch.linspace(0.0, 1.0, 200000):
        ro2_after = np.sqrt(ro2) * (1 + k1*ro2 + k2*ro2*ro2)
        map_dist.append([(1 + k1*ro2 + k2*ro2*ro2), ro2_after])
    map_dist = np.array(map_dist)
    print(map_dist)

    for z in torch.linspace(0.0, 1.0, 200000):
        z_after = np.sqrt(1 - z**2) / (z + mirror)
        z_dist.append([z, z_after])
    z_dist = np.array(z_dist)
    print(z_dist)

    map_dist = torch.from_numpy(map_dist)
    z_dist = torch.from_numpy(z_dist)

    def chunk(grid):
        x = grid[0, :]
        y = grid[1, :]

        x = (x - u0) / gamma1
        y = (y - v0) / gamma2
        dist = torch.sqrt(x*x + y*y)
        indx = torch.abs(map_dist[:, 1:] - dist[None, :]).argmin(dim=0)
        x /= map_dist[indx, 0]
        y /= map_dist[indx, 0]
    
        z_after = torch.sqrt(x*x + y*y)
        indx = torch.abs(z_dist[:, 1:] - z_after[None, :]).argmin(dim=0)

        x *= (z_dist[indx, 0] +mirror)
        y *= (z_dist[indx, 0] +mirror)

        xy = torch.stack((x, y))
        return xy

    xys = []
    for i in range(1400):
        xy = chunk(grid[:, i*1400:(i+1)*1400])
        xys.append(xy.permute(1, 0))
        if i % 10 == 0:
            print(i)
    xys = torch.cat(xys, dim=0)
    
    z = torch.sqrt(1. - torch.norm(xys, dim=1, p=2) ** 2)
    isnan = z.isnan()
    z[isnan] = 1.
    pcd = torch.cat((xys, z[:, None], isnan[:, None]), dim=1)
    print("saving grid")
    np.save('grid_fisheye_03.npy', pcd.detach().cpu().numpy())

    # show error map
    x, y, d = camera.cam2image(pcd[:, :3].permute(1, 0))

    error = (x - grid[0]) ** 2 + (y - grid[1]) ** 2

    error_map = error.reshape(1400, 1400).detach().cpu().numpy()
    error_map = np.clip(error_map, 0, 30)
    plt.imshow(error_map)
    plt.show()

