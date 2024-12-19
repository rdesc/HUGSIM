#!/bin/zsh
cuda=$1
out=$2

export CUDA_VISIBLE_DEVICES=$cuda

arr=("front" "front_left" "front_right" "left" "right" "back")
for cam in ${arr[@]}
do
    echo ${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/${cam}_camera \
    --output_dir ${out}/semantics/${cam}_camera \
    --model_path /nas/users/hyzhou/model_zoo/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
