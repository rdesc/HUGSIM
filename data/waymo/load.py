import os
import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
import json
import cv2
from scipy.spatial.transform import Rotation as SCR
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

opengl2waymo = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', type=str, required=True)
    parser.add_argument('-s', '--segment', type=str, required=True)
    parser.add_argument('-c', '--cameras', nargs='+', type=int, required=True)
    parser.add_argument('-o', '--outpath', type=str, required=True)
    parser.add_argument('--downsample', type=float, default=2)
    return parser.parse_args()

def roty_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def get_vertices(dim, bottom_center=np.array([0.0 ,0.0 ,0.0 ])):
    '''
    dim: length, height, width
    bottom_center: center of bottom face of 3D bounding box

    return: vertices of 3D bounding box (8*3)
    '''
    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2 
    vertices[[0,1,4,5], 1] = vertices[[0,1,4,5], 1]
    vertices[[2,3,6,7], 1] = vertices[[2,3,6,7], 1] - dim[1]
    vertices[[0,2,5,7], 2] = vertices[[0,2,5,7], 2] + dim[2] / 2
    vertices[[1,3,4,6], 2] = vertices[[1,3,4,6], 2] - dim[2] / 2

    return vertices

camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

if __name__ == '__main__':
    args = get_opts()

    seq_path = os.path.join(args.base_path, f"{args.segment}")
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    
    # create folders
    save_dir = args.outpath
    os.makedirs(save_dir, exist_ok=True)
    cams = args.cameras
    for cam in cams:
        os.makedirs(os.path.join(save_dir, "images", f"cam_{cam}"), exist_ok=True)
        
    ##########################################################################
    #                        read first frame info                         #
    #           lidar is only used for extracting camera height          #
    ##########################################################################
    for frame in datafile:
        lidar_points = []
        for laser_name in [dataset_pb2.LaserName.TOP, dataset_pb2.LaserName.FRONT, dataset_pb2.LaserName.SIDE_LEFT, dataset_pb2.LaserName.SIDE_RIGHT, dataset_pb2.LaserName.REAR]:
            laser = utils.get(frame.lasers, laser_name)
            laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
            range_images, camera_projections, range_image_top_pose = utils.parse_range_image_and_camera_projection(laser)
            points, _ = utils.project_to_pointcloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                laser_calibration
            )
            lidar_points.append(points[:, :3]) # in ego pose
        lidar_points = np.concatenate(lidar_points)
        ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (np.abs(lidar_points[:, 1]) < 3)
        lidar_points = lidar_points[ground_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)
        o3d.io.write_point_cloud(os.path.join(args.outpath, 'ground_lidar.ply'), pcd)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
        a, b, c, d = plane_model
        
        # camera calib
        all_campose = {}
        for camera in frame.context.camera_calibrations:
            if camera.name not in args.cameras:
                continue
            c2v = np.array(camera.extrinsic.transform).reshape(4,4)
            all_campose[camera.name] = c2v @ opengl2waymo
        front_cam_t = all_campose[1][:3, 3]
        height = -(a*front_cam_t[0] + b*front_cam_t[1] + d) / c
        front_cam_info = {
            "height": front_cam_t[2] - height,
            "rect_mat": None,
        }
        with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
            json.dump(front_cam_info, f, indent=2)
        
        break

    ##########################################################################
    #                     Read all frames infos                         #
    ##########################################################################
    ego_poses, extr, c2ws, intr, imsize, vehicles, dynamics = {}, {}, {}, {}, {}, {}, {}
    timestamps = []
    start_timestamp =  None
    for frame_idx, frame in tqdm(enumerate(datafile)):
        
        if start_timestamp is None:
            start_timestamp = frame.timestamp_micros / 1e6
        t = frame.timestamp_micros / 1e6 - start_timestamp
        
        timestamps.append(t)
        
        # Get Lidar calibration
        l2v = None
        for calibration in frame.context.laser_calibrations:
            if calibration.name == dataset_pb2.LaserName.TOP:
                l2v = np.array(calibration.extrinsic.transform).reshape(4,4)
                
        # images
        for img_pkg in frame.images:
            if img_pkg.name not in cams:
                continue
            img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if args.downsample > 1:
                h = int(h // args.downsample)
                w = int(w // args.downsample)
                img = cv2.resize(img, (w, h))
            output_path = os.path.join(save_dir, "images", f"cam_{img_pkg.name}", f"{str(frame_idx).zfill(6)}.png")
            cv2.imwrite(output_path, img)
            if img_pkg.name not in imsize:
                imsize[img_pkg.name] = []
            imsize[img_pkg.name].append((h, w))
            
            v2w = np.array(img_pkg.pose.transform).reshape(4, 4) # v2w at each camera timestamp
            if img_pkg.name not in extr:
                ego_poses[img_pkg.name] = []
            ego_poses[img_pkg.name].append(v2w)
            
         # camera calib
        for camera in frame.context.camera_calibrations:
            if camera.name not in cams:
                continue
            if camera.name not in extr:
                extr[camera.name] = []
            if camera.name not in intr:
                intr[camera.name] = []

            cam_intrinsic = np.eye(4)
            cam_intrinsic[0, 0] = camera.intrinsic[0] / args.downsample
            cam_intrinsic[1, 1] = camera.intrinsic[1] / args.downsample
            cam_intrinsic[0, 2] = camera.intrinsic[2] / args.downsample
            cam_intrinsic[1, 2] = camera.intrinsic[3] / args.downsample
            intr[camera.name].append(cam_intrinsic)

            c2v = np.array(camera.extrinsic.transform).reshape(4,4)
            extr[camera.name].append(c2v)
            
        # ego pose
        v2w = np.array(frame.pose.transform).reshape(4,4)

        # 3d bbox
        for obj in frame.laser_labels:
            type_name = type_list[obj.type]
            height = obj.box.height  # up/down
            width = obj.box.width  # left/right
            length = obj.box.length  # front/back
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2
            t_b2l = np.array([x,y,z,1]).reshape((4,1))
            t_b2w = v2w @ t_b2l
            rotation_y = -obj.box.heading - np.pi / 2
            if type_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                if obj.id not in vehicles:
                    vehicles[obj.id] = {
                        "rt": [],
                        "timestamp": [],
                        "frame": [],
                    }
                vehicles[obj.id]['rt'].append(np.array(t_b2w[:3, 0].tolist() + [length, height, width, rotation_y]))
                vehicles[obj.id]["timestamp"].append(t)
                vehicles[obj.id]['frame'].append(frame_idx)

    # normalize poses
    for cam, v2ws in ego_poses.items():
        for i, v2w in enumerate(v2ws):
            c2v = extr[cam][i]
            c2w = v2w @ c2v @ opengl2waymo
            if cam not in c2ws:
                c2ws[cam] = []
            c2ws[cam].append(c2w)
    inv_pose = np.linalg.inv(c2ws[1][0])
    for cam, poses in c2ws.items():
        poses = np.stack(poses)
        poses = np.einsum('njk,ij->nik', poses, inv_pose)
        c2ws[cam] = poses
    
    # filter dynamic vehicles
    dynamic_id = 0
    for objid, infos in vehicles.items():
        infos['rt'] = np.stack(infos['rt'])
        trans = infos['rt'][:, :3]
        trans = np.einsum('njk,ij->nik', trans[..., None], inv_pose[:3, :3])
        trans = trans[..., 0] + inv_pose[:3, 3]
        movement = np.max(np.max(trans, axis=0) - np.min(trans, axis=0))
        if movement > 1:
            infos["rt"][:, :3] = trans
            dynamics[dynamic_id] = infos
            dynamic_id += 1

    # post process dynamic infos
    verts, rts = {}, {}
    for dynamic_id, infos in dynamics.items():
        lhw = np.array(infos['rt'][0, 3:6])
        points = get_vertices(lhw)
        trans = infos['rt'][:, 0:3]
        roty = infos['rt'][:, 6]
        seq_visible = False
        for idx, fid in enumerate(infos['frame']):
            rt = np.eye(4)
            cam_roty = SCR.from_matrix(extr[1][fid][:3, :3]).as_euler('yxz')[0]
            rt[:3, :3] = roty_matrix(roty[idx] + cam_roty)
            rt[:3, 3] = trans[idx]
            
            points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            frame_visible = False
            for cam in cams:
                c2w = extr[cam][fid]
                w2c = np.linalg.inv(c2w)
                K = intr[cam][fid]
                h, w = imsize[cam][fid]
                points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]
                points_screen = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
                points_uv = (points_screen[:, :2] / points_screen[:, 2][:, None]).astype(int)
                valid_mask = (points_screen[:, 2] > 0) & (points_uv[:, 0] >= 0) & (points_uv[:, 1] >= 0) & (points_uv[:, 0] < w) & (points_uv[:, 1] < h)
                if np.sum(valid_mask) > 0:
                    frame_visible = True
                    seq_visible = True
                    break
                
            if frame_visible:
                if fid not in rts:
                    rts[fid] = {}
                rts[fid][dynamic_id] = rt.tolist()
                
        if seq_visible:
            verts[dynamic_id] = points.tolist()

    # write meta_data.json
    meta_data = {
        "camera_model": "OPENCV",
        "frames": [],
        "verts": verts,
        "inv_pose": inv_pose.tolist()
    }

    for i in range(len(intr[1])):
        for cam in cams:
            intrinsic = intr[cam][i]
            camtoworld = c2ws[cam][i]
            h, w = imsize[cam][i]
            info = {
                'rgb_path': f'./images/cam_{cam}/{str(i).zfill(6)}.png',
                'camtoworld': camtoworld.tolist(),
                'intrinsics': intrinsic.tolist(),
                'width': w,
                'height': h,
                'timestamp': timestamps[i],
                "dynamics": rts.get(i, {})
            }
            meta_data['frames'].append(info)

    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)