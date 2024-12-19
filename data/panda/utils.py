import numpy as np
import pyquaternion
import cv2
import os

ALLOWED_RIGID_CLASSES = (
    "Car",
    "Pickup Truck",
    "Medium-sized Truck",
    "Semi-truck",
    "Towed Object",
    "Motorcycle",
    "Other Vehicle - Construction Vehicle",
    "Other Vehicle - Uncommon",
    "Other Vehicle - Pedicab",
    "Emergency Vehicle",
    "Bus",
    "Personal Mobility Device",
    "Motorized Scooter",
    "Bicycle",
    "Train",
    "Trolley",
    "Tram / Subway",
)

ALLOWED_NONRIGID_CLASSES = (
    "Pedestrian",
    "Pedestrian with Object",
)

AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")
cameras = [cam + "_camera" for cam in AVAILABLE_CAMERAS]

def _pandaset_pose_to_matrix(pose):
    translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
    quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
    pose = np.eye(4)
    pose[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
    pose[:3, 3] = translation
    return pose

def _yaw_to_rotation_matrix(yaw: np.ndarray):
    """Converts array of yaw angles to rotation matrices."""
    rotation_matrices = np.zeros((yaw.shape[0], 3, 3))
    rotation_matrices[:, 0, 0] = np.cos(yaw)
    rotation_matrices[:, 0, 1] = -np.sin(yaw)
    rotation_matrices[:, 1, 0] = np.sin(yaw)
    rotation_matrices[:, 1, 1] = np.cos(yaw)
    rotation_matrices[:, 2, 2] = 1
    return rotation_matrices

def get_vertices(dim, bottom_center=np.array([0.0 ,0.0 ,0.0 ])):
    '''
    dim: length, height, width
    bottom_center: center of bottom face of 3D bounding box

    return: vertices of 3D bounding box (8*3)
    '''
    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2 
    vertices[[0,1,4,5], 1] = vertices[[0,1,4,5], 1] + dim[1] / 2
    vertices[[2,3,6,7], 1] = vertices[[2,3,6,7], 1] - dim[1] / 2
    vertices[[0,2,5,7], 2] = vertices[[0,2,5,7], 2] + dim[2] / 2
    vertices[[1,3,4,6], 2] = vertices[[1,3,4,6], 2] - dim[2] / 2

    return vertices

def point_in_bbox(vps, points):
    i, j, k, v = vps[1] - vps[0], vps[2] - vps[0], vps[5] - vps[0], points - vps[0]
    vi = np.dot(v, i)
    vj = np.dot(v, j)
    vk = np.dot(v, k)
    mask1 = (0 < vi) & (vi <= np.dot(i, i))
    mask2 = (0 < vj) & (vj <= np.dot(j, j))
    mask3 = (0 < vk) & (vk <= np.dot(k, k))
    mask = mask1 & mask2 & mask3
    return mask

def load_cam(sequence, camera, i, inv_pose, downsample=1):
    curr_cam = sequence.camera[camera]
    im_path = curr_cam._data_structure[i]
    im = cv2.imread(im_path)
    if camera == 'back_camera':
        im = im[:-250, ...]
    height = im.shape[0] // downsample
    width = im.shape[1] // downsample
    im = cv2.resize(im, (width, height))
    im_name = os.path.basename(im_path)

    pose = inv_pose @ _pandaset_pose_to_matrix(curr_cam.poses[i])
    intrinsic_ = curr_cam.intrinsics
    intrinsic = np.eye(4)
    intrinsic[0,0] = intrinsic_.fx / downsample
    intrinsic[1,1] = intrinsic_.fy / downsample
    intrinsic[0,2] = intrinsic_.cx / downsample
    intrinsic[1,2] = intrinsic_.cy / downsample

    timestamp = curr_cam.timestamps[i]

    return im, im_name, height, width, intrinsic, pose, timestamp