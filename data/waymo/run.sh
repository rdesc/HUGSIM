#!/bin/bash

cuda=4
export CUDA_VISIBLE_DEVICES=$cuda

# base_dir="/nas/datasets/Waymo_NOTR/static"
# segment="segment-16608525782988721413_100_000_120_000_with_camera_labels.tfrecord"

base_dir="/nas/datasets/Waymo_NOTR/dynamic"
segment="segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord"

seg_prefix=$(echo $segment| cut -c 9-15)
seq_name=${seg_prefix}
out=/data3/hyzhou/data/HUGSIM/release/waymo/$seq_name
cameras="1 2 3"


mkdir -p $out

# load images, camera pose, etc
python waymo/load.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type waymo
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype waymo