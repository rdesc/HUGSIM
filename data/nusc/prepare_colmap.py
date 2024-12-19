import os
import numpy as np
from pathlib import Path
import argparse
import json
from colmap.colmap import COLMAPAuto, rotmat2qvec


def get_opts():
    parser = argparse.ArgumentParser("colmap prepare", description='prepare colamp image dataset')
    parser.add_argument('-i', '--in_path', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    with open(os.path.join(args.in_path, 'meta_data.json'), 'r') as jf:
        meta_data = json.load(jf)

    path_prior = os.path.join(args.in_path, 'prior')
    path_rigid = os.path.join(args.in_path, 'cam_rigid_config.json')
    os.makedirs(path_prior, exist_ok=True)

    # points3D
    Path(os.path.join(path_prior, 'points3D.txt')).touch()

    with open(os.path.join(path_prior, 'cameras.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames'][:6]):
            intr = np.array(frame['intrinsics'])
            w, h = frame['width'], frame['height']
            intr4 = [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]
            intr4 = [str(i.item()) for i in intr4]
            str_intr = ' '.join(intr4)
            f.write(f"{idx+1} PINHOLE {w} {h} {str_intr}" + '\n')

    # images
    with open(os.path.join(path_prior, 'images.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames']):
            c2w = np.array(frame['camtoworld'])
            img_path = frame['rgb_path']

            rel_path = os.path.relpath(img_path, "./images")

            w2c = np.linalg.inv(c2w)
            q_w2c = [str(v.item()) for v in rotmat2qvec(w2c[:3, :3])]
            t_w2c = [str(v.item()) for v in w2c[:3, -1]]
            cam_id = idx % 6 + 1
            line = f"{idx+1} {' '.join(q_w2c)} {' '.join(t_w2c)} {cam_id} {rel_path}"
            f.write(line + '\n\n')

    auto = COLMAPAuto(args.in_path)

    auto.feature_extract()
    auto.sequential_matcher()
    auto.point_triangulator()
    os.system(f'cp -r {auto.path_tri} {auto.path_ba}')
    auto.rigid_ba(path_rigid)
    auto.point_triangulator_ba()