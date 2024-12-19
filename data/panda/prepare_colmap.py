import numpy as np
import os
from pathlib import Path
import argparse
import json
import sqlite3

class COLMAPAuto:
    def __init__(self, dense):
        self.path_dense = dense
        self.path_database = os.path.join(self.path_dense, 'database.db')
        self.path_images = os.path.join(self.path_dense, 'images')
        self.path_masks = os.path.join(self.path_dense, 'dynamics')
        # self.path_pose = os.path.join(self.path_dense, 'poses.txt')
        # self.path_project = os.path.join(self.path_dense, 'auto.ini')
        self.path_prior = os.path.join(self.path_dense, 'prior')
        self.path_tri = os.path.join(self.path_dense, 'colmap_sparse')  # tri'
        self.path_ba = os.path.join(self.path_dense, 'colmap_sparse')
        self.path_sparse = os.path.join(self.path_dense, 'colmap_sparse')
        self.file_rescale = os.path.join(self.path_dense, 'colmap2world.json')

    # @property
    # def database(self):
    #     db = cdb.COLMAPDatabase.connect(self.path_database)
    #     return db

    def remove_database(self):
        if os.path.exists(self.path_database):
            print(f'remove exist database: {self.path_database}')
            os.remove(self.path_database)

    def __call__(self, args):
        script = ' '.join(args)
        print(20 * '=')
        print(script)
        print(20 * '=')
        os.system(script)

    def feature_extract(self, cam_calib_in=None):
        self.remove_database()

        self([
            'colmap', 'feature_extractor',
            '--database_path', self.path_database,
            '--image_path', self.path_images,
            # '--ImageReader.camera_params', cam_calib_in,
            '--ImageReader.mask_path', self.path_masks,
            '--ImageReader.single_camera_per_folder', '1',
            '--ImageReader.camera_model', 'PINHOLE ',
            '--SiftExtraction.use_gpu', '1',
            # '--SiftExtraction.max_num_features', '8192',
            # '--SiftExtraction.estimate_affine_shape', '0',
            '--SiftExtraction.upright', '1',
            # '--SiftExtraction.domain_size_pooling', '0'
        ])

    def sequential_matcher(self):
        if not os.path.exists(self.path_database):
            assert FileNotFoundError(f'database file required')

        self([
            'colmap', 'sequential_matcher',
            '--database_path', self.path_database,
            '--SequentialMatching.overlap', '20',
            '--SiftMatching.use_gpu', '1',  # use gpu or not
        ])

    def vocab_matcher(self, vocab_tree_path):
        if not os.path.exists(self.path_database):
            assert FileNotFoundError(f'database file required')

        self([
            'colmap', 'vocab_tree_matcher',
            '--database_path', self.path_database,
            '--VocabTreeMatching.vocab_tree_path', vocab_tree_path,
            '--SiftMatching.guided_matching', '1'
        ])

    def point_triangulator(self):
        os.makedirs(self.path_tri, exist_ok=True)

        self([
            'colmap', 'point_triangulator',
            '--database_path', self.path_database,
            '--image_path', self.path_images,
            '--input_path', self.path_prior,
            '--output_path', self.path_tri,
        ])

    def id_mapping_in_database(self):
        dbconn = sqlite3.connect(self.path_database)
        cur = dbconn.cursor()
        name__id = {}
        for image_id, image_name in cur.execute('''SELECT image_id, name FROM images '''):
            name__id[image_name] = image_id
        return name__id

    def ba(self, inter=0):
        os.makedirs(self.path_ba, exist_ok=True)

        self([
            'colmap', 'bundle_adjuster',
            '--input_path', self.path_tri,
            '--output_path', self.path_ba,
            '--BundleAdjustment.max_num_iterations', '10',
            '--BundleAdjustment.refine_focal_length', '1',
            '--BundleAdjustment.refine_principal_point', '1',
            '--BundleAdjustment.refine_extra_params', '1',
            '--BundleAdjustment.refine_extrinsics', '1'
        ])

    def mapper(self, max_error=12.0):
        if not os.path.exists(self.path_database):
            assert FileNotFoundError(f'database file required')

        if os.path.exists(self.path_sparse):
            os.system(f'rm -rf {self.path_sparse}')
        os.makedirs(self.path_sparse, exist_ok=True)

        self([
            'colmap', 'mapper',
            '--database_path', self.path_database,
            '--image_path', self.path_images,
            # '--input_path', self.path_ba,
            '--output_path', self.path_sparse,
            '--Mapper.init_max_error', str(max_error),
            '--Mapper.fix_existing_images', '0',
            '--Mapper.ba_global_max_num_iterations', '10',
            '--Mapper.ba_global_max_refinements', '1',
            '--Mapper.tri_ignore_two_view_tracks', '0'
        ])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_opts():
    parser = argparse.ArgumentParser("colmap prepare", description='prepare colamp image dataset')
    parser.add_argument('-i', '--in_path', type=str, required=True)
    parser.add_argument('-c', '--cameras', nargs='+', type=int, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()

    with open(os.path.join(args.in_path, 'meta_data.json'), 'r') as jf:
        meta_data = json.load(jf)

    path_prior = os.path.join(args.in_path, 'prior')
    os.makedirs(path_prior, exist_ok=True)

    # points3D
    Path(os.path.join(path_prior, 'points3D.txt')).touch()

    # cameras
    with open(os.path.join(path_prior, 'cameras.txt'), 'w') as f:
        for idx, frame in enumerate(meta_data['frames'][:len(args.cameras)]):
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
            cam_id = idx % len(args.cameras) + 1
            line = f"{idx+1} {' '.join(q_w2c)} {' '.join(t_w2c)} {cam_id} {rel_path}"
            f.write(line + '\n\n')

    auto = COLMAPAuto(args.in_path)

    auto.feature_extract()
    auto.sequential_matcher()
    # auto.mapper()
    auto.point_triangulator()