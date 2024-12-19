#!/bin/zsh
cuda=$1
out=$2

export CUDA_VISIBLE_DEVICES=$cuda

arr=("FRONT" "FRONT_LEFT" "FRONT_RIGHT" "BACK_LEFT" "BACK_RIGHT" "BACK")
for cam in ${arr[@]}
do
    echo CAM_${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/CAM_${cam} \
    --output_dir ${out}/semantics/CAM_${cam} \
    --model_path /data3/hyzhou/model_zoo/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
