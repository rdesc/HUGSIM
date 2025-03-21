####### NOTE: probably best to execute line by line rather than trying to run this as a script #######

export HUGSIM_WORKSPACE=$HOME/hugsim_workspace
mkdir -p $HUGSIM_WORKSPACE

cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/HUGSIM/
cd HUGSIM
# Setup hugsim conda env
conda create --name hugsim python=3.11
conda activate hugsim
# from Mila IT https://mila-umontreal.slack.com/archives/CFAS8455H/p1740539055496489?thread_ts=1740537369.229409&cid=CFAS8455H
conda config --set always_softlink False; conda config --set allow_softlinks False
# first install pytorch before installing rest of dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
conda env update --name hugsim --file conda_env.yaml

# Install hug_sim Gymnasium environment for closed-loop simulation
cd $HUGSIM_WORKSPACE/HUGSIM/sim
pip install -e .

# Fix for UniDepth
# option 1 (easy)
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git@main#subdirectory=unidepth/ops/knn
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git@main#subdirectory=unidepth/ops/extract_patches
# option 2 (if above does not work)
# cd $HUGSIM_WORKSPACE
# git clone https://github.com/rdesc/UniDepth
# cd UniDepth/unidepth/ops/knn/
# ./compile.sh
# cd ../extract_patches/
# ./compile.sh
# Build xFormers for pytorch 2.5.1+cu118 and python 3.11.11
# (optional) Set TORCH_CUDA_ARCH_LIST env variable if running and building on different GPU types
pip install -v -U git+https://github.com/rdesc/xformers.git@pytorch_2.5.1#egg=xformers --index-url https://download.pytorch.org/whl/cu118
# NOTE: Getting this warning "Triton is not available, some optimizations will not be enabled."

# Download the sample data
cd $HUGSIM_WORKSPACE/HUGSIM/data
git clone https://huggingface.co/datasets/hyzhou404/HUGSIM/ sample_data
cd sample_data/sample_data
git lfs install
git lfs pull
unzip 3DRealCar.zip; unzip data.zip; unzip model.zip

# Download model weights for InverseForm
cd $HUGSIM_WORKSPACE/HUGSIM/data/InverseForm/checkpoints
wget https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/hrnet48_OCR_HMS_IF_checkpoint.pth
wget https://github.com/Qualcomm-AI-research/InverseForm/releases/download/v1.0/distance_measures_regressor.pth
# install InverseForm dependency 
pip install --no-build-isolation --config-settings=--build-option=--cpp_ext --config-settings=--build-option=--cuda_ext git+https://github.com/NVIDIA/apex.git

# Install clients (need to create a separate conda env)
cd $HUGSIM_WORKSPACE
conda deactivate
# 1. UniAD
# instructions are taken from https://github.com/OpenDriveLab/UniAD/blob/v2.0/docs/INSTALL.md
conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
git checkout v1.6.0
export MMCV_WITH_OPS=1 MMCV_CUDA_ARGS=-std=c++17
pip install -v -e .
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6

cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/UniAD_SIM  # a fork of UniAD which contains hugsim integration
cd UniAD_SIM
pip install -r requirements.txt

# download model weights
mkdir ckpts && cd ckpts
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth

# download UniAD data info
mkdir -p data/infos
cd data/infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos
cd ..
mkdir others
cd others
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl

# 2. VAD (not tested)
cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/VAD_SIM  # a fork of VAD which contains hugsim integration
cd VAD_SIM
# instructions are taken from https://github.com/hustvl/VAD/blob/main/docs/install.md
# (env can be same conda env as UniAD and therefore most of the steps are already done during UniAD setup)

# download model weights
mkdir ckpts && cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
pip install gdown
gdown https://drive.google.com/uc?id=1KgCC_wFqPH0CQqdr6Pp2smBX5ARPaqne
gdown https://drive.google.com/uc?id=1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS

# 3. LTF (Latent TransFuser) + NAVSIM (not tested)
cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/NAVSIM
cd NAVSIM
# instructions are taken from https://github.com/autonomousvision/navsim/blob/main/docs/install.md
conda env create --name navsim -f environment.yml
conda activate navsim
pip install -e .

# Data preparation
# 1. nuScenes
# follow instructions at https://www.nuscenes.org/nuscenes#download
export NUPLAN_DATA_ROOT=~/scratch/nuscenes/
# Upsample from 2 Hz to 12 Hz (2 Hz is too sparse to reconstruct with colmap???)
cd $HUGSIM_WORKSPACE
git clone https://github.com/JeffWang987/ASAP
cd ASAP
mkdir out
# activate the conda env which should already have the dependencies installed for ASAP
conda activate uniad2.0
# fix bug in nusc_annotation_generator.py by commenting out a line
sed -i '/nusc_20Hz_rst = mmcv.load(opts.lidar_inf_rst_path)/s/^/    #/' sAP3D/nusc_annotation_generator.py
python -m sAP3D.nusc_annotation_generator \
    --data_path $NUPLAN_DATA_ROOT \
    --data_version v1.0-trainval \
    --ann_frequency 12 \
    --ann_strategy interp
# NOTE: the 12Hz annotations are calculated by the object interpolation, NOT object interpolation + temporal database 
# unclear whether just object interpolation is sufficient....  https://github.com/JeffWang987/ASAP/blob/main/docs/prepare_data.md

# (optional) test reconstruction by modifying params as neccessary in the script data/nusc/run.sh and running
cd $HUGSIM_WORKSPACE/HUGSIM/data
bash ./nusc/run.sh

# (optional) test gaussian splat training
seq=scene-0103
input_path=~/scratch/hugsim_data/release/nusc/${seq}
output_path=~/scratch/hugsim_data/release/nusc/${seq}/outputs
dataset_name=nusc
mkdir -p ${output_path}

python -u train_ground.py --data_cfg ./configs/${dataset_name}.yaml \
        --source_path ${input_path} --model_path ${output_path}

python -u train.py --data_cfg ./configs/${dataset_name}.yaml \
        --source_path ${input_path} --model_path ${output_path}

# (optional) test closed loop simulation
# set the params for the scenario config (e.g. configs/benchmark/nuscenes/scene-0383-easy-00.yaml)
# set the params for the base config (e.g. configs/sim/nuscenes_base.yaml)
# set the params for client config UniAD_SIM/tools/e2e.sh
python closed_loop.py --scenario_path ./configs/benchmark/nuscenes/scene-0383-hard-00.yaml \
                        --base_path ./configs/sim/nuscenes_base.yaml \
                        --camera_path ./configs/sim/nuscenes_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad uniad \
                        --ad_cuda 0