#!/bin/zsh
cuda=$1
out=$2

echo $cuda
echo $out
export CUDA_VISIBLE_DEVICES=$cuda

arr=("1" "2" "3")
for cam in ${arr[@]}
do
    echo cam_${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/cam_${cam} \
    --output_dir ${out}/semantics/cam_${cam} \
    --model_path /data3/hyzhou/model_zoo/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
