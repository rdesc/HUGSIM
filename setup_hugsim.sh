#####################################################

# option 1: build from source in a new conda env
conda create --name colmap python=3.9
conda activate colmap

module load cudatoolkit/11.8  # mila cluster, for enabling cuda

conda install -c conda-forge -y cmake ninja libboost boost eigen flann freeimage metis glog gtest gmock sqlite glew qt libopengl pyqt icu sip cgal ceres-solver freeglut mesa mesalib gcc_linux-64=10 gxx_linux-64=10

git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_INSTALL_PREFIX=$HOME/colmap_install -DCMAKE_VERBOSE_MAKEFILE=TRUE -DOPENGL_opengl_LIBRARY=$HOME/miniconda3/envs/colmap/lib/libOpenGL.so.0 -DOPENGL_glx_LIBRARY=$HOME/miniconda3/envs/colmap/lib/libGLX.so.0  # need to point to the opengl binaries
ninja
ninja install

# option 2: install from conda
conda install conda-forge::colmap

#####################################################

cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/HUGSIM/
cd HUGSIM

# Setup hugsim conda env
conda create --name hugsim python=3.11
conda activate hugsim
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
# Getting this warning:
# Triton is not available, some optimizations will not be enabled.

# Download the sample data
cd $HUGSIM_WORKSPACE/HUGSIM/data
git clone https://huggingface.co/datasets/hyzhou404/HUGSIM/ sample_data
cd sample_data/sample_data
git lfs install
git lfs pull
unzip 3DRealCar.zip; unzip data.zip; unzip model.zip

# Install clients (need to create a separate conda env)
conda deactivate
# 1. UniAD
cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/UniAD_SIM
# follow instructions at https://github.com/OpenDriveLab/UniAD/blob/v2.0/docs/INSTALL.md

# 2. VAD
cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/VAD_SIM
# follow instructions at https://github.com/hustvl/VAD/blob/main/docs/install.md (can be same conda env as UniAD)

# 3. LTF (Latent TransFuser) + NAVSIM
cd $HUGSIM_WORKSPACE
git clone https://github.com/hyzhou404/NAVSIM
# follow instructions at https://github.com/autonomousvision/navsim/blob/main/docs/install.md


# TODO: set the right paths for the models in the client codes (we probably don't need to worry about paths for the datasets in the clients)
# Run HUGSIM demos
# download model weights where necessary