import open3d as o3d
import argparse
import os
import json

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outpath', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()
    pcd = o3d.io.read_point_cloud(os.path.join(args.outpath, "ground_lidar.ply"))
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
    a, b, c, d = plane_model
    with open(os.path.join(args.outpath, 'front_info.json'), 'r') as f:
        front_cam_info = json.load(f)
    front_cam_t = front_cam_info['front_cam_t']
    height = -(a*front_cam_t[0] + b*front_cam_t[1] + d) / c
    front_cam_info['height'] = front_cam_t[2] - height
    with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f)
