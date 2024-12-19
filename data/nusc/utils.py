import numpy as np
import pyquaternion
import os
import cv2
import torch

AVAILABLE_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

WLH_TO_LWH = np.array(
    [
        [0, 1.0, 0, 0],
        [-1.0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0],
    ]
)

WLH_TO_LWH2 = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)

ALLOWED_RIGID_CLASSES = (
    "vehicle.car",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.bus",
    "vehicle.bus",
    "vehicle.truck",
    "vehicle.trailer",
    "movable_object.pushable_pullable",
)
ALLOWED_DEFORMABLE_CLASSES = ("human.pedestrian",)
ALLOWED_CLASSES = ALLOWED_RIGID_CLASSES + ALLOWED_DEFORMABLE_CLASSES

def _rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix
    pose[:3, 3] = t_vec
    return pose

def find_all_sample_data(nusc, sample_data_token):
    """Finds all sample data from a given sample data token."""
    curr_token = sample_data_token
    sd = nusc.get("sample_data", curr_token)
    # Rewind to first sample data
    while sd["prev"]:
        curr_token = sd["prev"]
        sd = nusc.get("sample_data", curr_token)
    # Forward to last sample data
    all_sample_data = [sd]
    while sd["next"]:
        curr_token = sd["next"]
        try:
            sd = nusc.get("sample_data", curr_token)
        except KeyError:
            print(curr_token)
            continue
        all_sample_data.append(sd)
    return all_sample_data

def find_all_sample(nusc, sample):
    """Finds all samples from a given sample token."""
    samples = []
    while sample['next']:
        samples.append(sample)
        sample = nusc.get('sample', sample['next'])
    samples.append(sample)
    return samples

def frame_check(nusc, sample, cameras, dataroot, output):
    """check if the frame has all camera images"""
    for cam in cameras:
        sample_data = nusc.get('sample_data', sample['data'][cam])
        im_path = sample_data['filename']
        im_fn = os.path.basename(im_path)
        if not os.path.exists(os.path.join(dataroot, im_path)):
            return False
        if os.path.exists(os.path.join(output, 'images', cam, im_fn)):
            return False
    return True

def get_sample_pose(nusc, sd):
    """get camera pose of given sample data"""
    calibrated_sensor_data = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_pose_data = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_pose = _rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
    cam_pose = _rotation_translation_to_pose(
        calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
    )
    return ego_pose @ cam_pose, calibrated_sensor_data["camera_intrinsic"]

def load_cam(nusc, sample_data, inv_pose, dataroot, downsample=1):
    """load cam image and camera parameters"""
    extrinsic, intr = get_sample_pose(nusc, sample_data)
    pose = inv_pose @ extrinsic
    im_path = os.path.join(dataroot, sample_data["filename"])
    im = cv2.imread(im_path)
    # Crop 80+66 for scene-0164, following NeuRAD
    if sample_data['channel'] == 'CAM_BACK':
        im = im[:-80, ...]
    height = int(im.shape[0] // downsample)
    width = int(im.shape[1] // downsample)
    im = cv2.resize(im, (width, height))
    im_name = os.path.basename(im_path)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = intr[0][0] / downsample
    intrinsic[1, 1] = intr[1][1] / downsample
    intrinsic[0, 2] = intr[0][2] / downsample
    intrinsic[1, 2] = intr[1][2] / downsample

    return im, im_name, height, width, intrinsic, pose

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

def is_label_allowed(label, allowed_classes):
    """Check if label is allowed, on all possible hierarchies."""
    split_label = label.split(".")
    for i in range(len(split_label)):
        if ".".join(split_label[: i + 1]) in allowed_classes:
            return True
    return False

def traj_dict_to_list(traj):
    """Convert a dictionary of lists with trajectories to a list of dictionaries with trajectories"""
    allowed_classes = set(ALLOWED_CLASSES)
    traj_out = []
    for instance_token, traj_list in traj.items():
        poses = torch.from_numpy(np.stack([t["pose"] for t in traj_list]).astype(np.float32))
        times = torch.from_numpy(np.array([t["time"] for t in traj_list]))
        dims = torch.from_numpy(np.array([t["wlh"] for t in traj_list]).astype(np.float32))
        dims = dims.max(0).values  # take max dimensions (important for deformable objects)
        dynamic = (poses[:, :3, 3].std(dim=0) > 0.50).any()
        stationary = not dynamic  # TODO: maybe make this stricter
        if stationary or not is_label_allowed(traj_list[0]["label"], allowed_classes):
            continue
        traj_dict = {
            "uuid": instance_token,
            "label": traj_list[0]["label"],
            "poses": poses,
            "timestamps": times,
            "dims": dims,
            "stationary": stationary,
            "symmetric": "human" not in traj_list[0]["label"],
            "deformable": "human" in traj_list[0]["label"],
        }
        traj_out.append(traj_dict)
    return traj_out

def get_box(nusc, box_token, inv_pose, c2w=None, updated_c2w=None, only_vehicle=False):
    box = nusc.get_box(box_token)
    instance_token = nusc.get("sample_annotation", box.token)["instance_token"]
    if only_vehicle and 'vehicle' not in box.name:
        return None, None, None
    pose = np.eye(4)
    pose[:3, :3] = box.orientation.rotation_matrix
    pose[:3, 3] = np.array(box.center)

    if c2w is None:
        pose = inv_pose @ pose @ WLH_TO_LWH
        lhw = np.array(box.wlh)
    else:
        pose = updated_c2w @ np.linalg.inv(c2w) @ inv_pose @ pose @ WLH_TO_LWH
        lhw = np.array(box.wlh)

    return instance_token, pose, lhw
