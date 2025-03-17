import argparse
import numpy as np
import os
import pickle
from flask import Flask, render_template, request, jsonify, send_from_directory
from scipy.spatial.transform import Rotation as SCR
from glob import glob

app = Flask(__name__)

parser = argparse.ArgumentParser(description="Flask app with dynamic scene and car selection")
parser.add_argument("--scene", type=str, required=True, help="Path to the scene file")
parser.add_argument("--car_folder", type=str, required=True, help="Path to the car directories")
args = parser.parse_args()

splat_file = os.path.join(args.scene, 'vis', 'scene.splat')
assert os.path.exists(splat_file), f"Scene file {splat_file} does not exist, please check."
SCENE_FILE_PATH = splat_file
print(SCENE_FILE_PATH)
SCENE_FILE_DIR = os.path.dirname(SCENE_FILE_PATH)
SCENE_FILE_NAME = os.path.basename(SCENE_FILE_PATH)
SMT_FILE_NAME = 'semantic.ply'

CAR_FOLDER_PATH = args.car_folder
CAR_FILES = [os.path.basename(splat) for splat in glob(os.path.join(CAR_FOLDER_PATH, '*.splat'))]

@app.route('/')
def index():
    return render_template('index.html',scene_file = SCENE_FILE_NAME, car_files=CAR_FILES, smt_file = SMT_FILE_NAME)

@app.route('/scene/<path:filename>')
def serve_scene_file(filename):
    return send_from_directory(SCENE_FILE_DIR, filename)

@app.route('/smt/<path:filename>')
def serve_semanticfile(filename):
    return send_from_directory(SCENE_FILE_DIR, filename)

@app.route('/car/<path:filename>')
def serve_car_file(filename):
    return send_from_directory(CAR_FOLDER_PATH, filename)

@app.route('/get_height', methods=['POST'])
def get_height():
    data = request.json
    x = data.get('x')
    z = data.get('z')
    try:
        y = calculate_height(x, z)
        return jsonify({"y": y})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def dense_cam_poses(cam_poses, cmds):
    for i in range(4):
        dense_poses = []
        dense_cmds = []
        for i in range(cam_poses.shape[0]-1):
            cam1 = cam_poses[i]
            cam2 = cam_poses[i+1]
            dense_poses.append(cam1)
            dense_cmds.append(cmds[i])
            if np.linalg.norm(cam1[:3, 3]-cam2[:3, 3]) > 0.2:
                euler1 = SCR.from_matrix(cam1[:3, :3]).as_euler("XYZ")
                euler2 = SCR.from_matrix(cam2[:3, :3]).as_euler("XYZ")
                interp_euler = (euler1 + euler2) / 2
                interp_trans = (cam1[:3, 3] + cam2[:3, 3]) / 2
                interp_pose = np.eye(4)
                interp_pose[:3, :3] = SCR.from_euler("XYZ", interp_euler).as_matrix()
                interp_pose[:3, 3] = interp_trans
                dense_poses.append(interp_pose)
                dense_cmds.append(cmds[i])
        dense_poses.append(cam_poses[-1])
        dense_poses = np.stack(dense_poses)
        cam_poses = dense_poses
        cmds = dense_cmds
        
    return cam_poses, cmds

def calculate_height(u, v):
    with open(os.path.join(args.scene, 'ground_param.pkl'), 'rb') as f:
        cam_poses, cam_heights, commands = pickle.load(f)
        cam_poses, commands = dense_cam_poses(cam_poses, commands)
    cam_dist = np.sqrt(
        (cam_poses[:, 0, 3] - u)**2 + (cam_poses[:, 2, 3] - v)**2
    )
    nearest_cam_idx = np.argmin(cam_dist, axis=0)
    nearest_c2w = cam_poses[nearest_cam_idx]

    nearest_w2c = np.linalg.inv(nearest_c2w)
    uhv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
    uhv_local[1] = 0
    uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
    return uhv_world[1]

if __name__ == '__main__':
    app.run(debug=False)