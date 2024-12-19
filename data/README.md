# Data Preparation

<details> <summary>KITTI-360</summary>

Download data from [KITTI-360 website](https://www.cvlibs.net/datasets/kitti-360/download.php). Perspective, fisheye images and calibrations are required. 

Please replace **\$\{seq\}**, **\$\{start\}**, **\$\{end\}** varibales to select slice of KITTI-360, replace **\$\{root\_dir\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./kitti360/run.sh
```
</details>

<details> <summary>Waymo Open Dataset</summary>

Download Waymo NOTR dataset following [the EmerNeRF doc](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md).

Please select the **\$\{segment\}**, replace **\$\{base\_dir\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./waymo/run.sh
```
</details>

<details> <summary>NuScenes</summary>

Download Nuscenes dataset from [nuScenes Website](https://www.nuscenes.org/nuscenes#download).

The original key frames in Nuscenes are 2Hz, which is too sparse to reconstruct. Please follow [ASAP](https://github.com/JeffWang987/ASAP/tree/52316629f2a87ef2ef5bbc634d33e9544b5e39a7) to convert sweep data as key frames. The output version of **ASAP** is **interp_12Hz_trainval**.

Please select the **\$\{seq\}**, replace **\$\{data\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./nusc/run.sh
```
</details>

<details> <summary>PandaSet</summary>

The PandaSet official download link is no longer available. PandaSet can still be downloaded from [Hugging face](https://huggingface.co/datasets/georghess/pandaset/tree/main), thanks Georg Hess for sharing!

Please select the **\$\{seq\}**, replace **\$\{data\}** and **\$\{out\}** variables as paths on your machine.

Run the following scripts to generate data for HUGSIM:
``` bash
cd data
zsh ./pandaset/run.sh
```
</details>
