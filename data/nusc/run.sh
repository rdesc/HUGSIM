#!/bin/zsh

export PYTHONPATH="${PWD}:$PYTHONPATH"

cuda=4
seq='scene-0655'
data='/nas/datasets/nuScenes/raw/Trainval'
version='interp_12Hz_trainval'
start=0
end=180
out=/data3/hyzhou/data/HUGSIM/release/nusc/${seq}

export CUDA_VISIBLE_DEVICES=$cuda

mkdir -p ${out}
python nusc/load.py --datapath ${data} --version ${version} --seq ${seq} --out ${out} \
        --start ${start} --end ${end} --downsample 2 --video

# generate semantic mask
cd InverseForm
./infer_nuscenes.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type nuscenes

# COLMAP sparse model
rm -rf ${out}/colmap_sparse*
rm ${out}/database.db*
rm -rf ${out}/prior
python nusc/prepare_colmap.py -i ${out}

echo "convert model into ply format"
colmap model_converter \
        --input_path ${out}/colmap_sparse_tri \
        --output_path ${out}/sparse_ba.ply \
        --output_type PLY

python colmap/update_campose.py --datapath ${out}

python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype nuscenes