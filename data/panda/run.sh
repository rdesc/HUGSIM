#!/bin/zsh

cuda=5
seq=027
data=/nas/datasets/pandaset
out=/data3/hyzhou/data/HUGSIM/release/pandaset/${seq}
cameras='0 1 2 3 4 5'

export CUDA_VISIBLE_DEVICES=$cuda

python panda/load.py --datapath ${data} --seq ${seq} --out ${out} --downsample 2 --video

# generate semantic mask
cd InverseForm
./infer_pandaset.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type pandaset
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype pandaset