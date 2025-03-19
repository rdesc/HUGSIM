#!/bin/bash

export PYTHONPATH="${PWD}:$PYTHONPATH"

cuda=0                          # CUDA_VISIBLE_DEVICES
seq='scene-0103'
data=$NUPLAN_DATA_ROOT          # nuScenes data root 
version='interp_12Hz_trainval'  
start=0                         # start frame index
end=-1                          # end frame index
out=~/scratch/hugsim_data/release/nusc/${seq}

# export CUDA_VISIBLE_DEVICES=$cuda

# confirm CUDA is available
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.backends.cudnn.version())"

mkdir -p ${out}

python nusc/load.py --datapath ${data} --version ${version} --seq ${seq} --out ${out} --start ${start} --end ${end} --downsample 2 --video

# generate semantic mask
cd InverseForm
./infer_nuscenes.sh ${cuda} ${out}

cd $HUGSIM_WORKSPACE/HUGSIM/data
python utils/create_dynamic_mask.py --data_path ${out} --data_type nuscenes  # TODO: what is this for?

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

# Get depth information using UniDepthV2
python utils/estimate_depth.py --out ${out}

python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype nuscenes