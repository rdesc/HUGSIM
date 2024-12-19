#!/bin/zsh

export PYTHONPATH="${PWD}:$PYTHONPATH"

cuda=5
root_dir="/nas/datasets/KITTI-360"
seq='0000'
start=5250
end=5350
pad_start=$(printf "%010d" "$start")
pad_end=$(printf "%010d" "$end")
out=/data3/hyzhou/data/HUGSIM/release/kitti360/${seq}_${start}_${end}
cameras="0 1 2 3"

export CUDA_VISIBLE_DEVICES=$cuda

mkdir -p ${out}/images/cam_0
mkdir -p ${out}/images/cam_1
mkdir -p ${out}/images/cam_2_fisheye
mkdir -p ${out}/images/cam_3_fisheye
mkdir -p ${out}/images/cam_2
mkdir -p ${out}/images/cam_3
cp ${root_dir}/2013_05_28_drive_${seq}_sync/image_00/data_rect/{${pad_start}..${pad_end}}.png ${out}/images/cam_0
cp ${root_dir}/2013_05_28_drive_${seq}_sync/image_01/data_rect/{${pad_start}..${pad_end}}.png ${out}/images/cam_1
cp ${root_dir}/2013_05_28_drive_${seq}_sync/image_02/data_rgb/{${pad_start}..${pad_end}}.png ${out}/images/cam_2_fisheye
cp ${root_dir}/2013_05_28_drive_${seq}_sync/image_03/data_rgb/{${pad_start}..${pad_end}}.png ${out}/images/cam_3_fisheye

python kitti360/mask_fisheye.py --out ${out}

# generate semantic mask
cd InverseForm
./infer_kitti360.sh ${cuda} ${out} '0 1'
./infer_kitti360_fish.sh ${cuda} ${out} '2 3'
cd -
echo $PWD

mkdir -p ${out}/semantics/cam_2
mkdir -p ${out}/semantics/cam_3

# # convert fish2persp and create metadata
python kitti360/load.py --root $root_dir --out $out --start $start --end $end --cams 0 1 2 3
rm -rf ${out}/images/cam_2_fisheye
rm -rf ${out}/images/cam_3_fisheye
rm -rf ${out}/semantics/cam_2_fisheye
rm -rf ${out}/semantics/cam_3_fisheye

python utils/create_dynamic_mask.py --data_path ${out} --data_type kitti360
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype kitti360