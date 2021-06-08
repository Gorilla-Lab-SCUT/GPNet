# Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps 
Pytorch implementation of [GPNet](https://arxiv.org/abs/2009.12606).

## Environment
- Ubuntu 16.04
- pytorch 0.4.1
- CUDA 8.0 or CUDA 9.2

Our depth images are saved in `.exr` files, please install the [OpenEXR](https://github.com/AcademySoftwareFoundation/openexr/blob/master/INSTALL.md), then run `pip install OpenEXR`.

## Build CUDA kernels
- ``cd lib/pointnet2``
- ``mkdir build && cd build``
- ``cmake .. && make``

## Dataset
Our dataset is available at [Google Driver](https://drive.google.com/file/d/1hZmQhuTrKRn8BMyAq-bI13rQSrdGQdJH/view?usp=sharing). [Backup](https://pan.baidu.com/s/1Gf0cIgaL1s30n22z7sOuRA) (2qln).  
$\color{red}{Warning!!!}$  
<font color='red'> The contact points in our released dataset are not correct, please run the following script to get the correct contact points.</font>  
``python get_contact_cos.py``


## Simulation Environment
The simulation environment is built on [PyBullet](https://pybullet.org/wordpress/). You can use `pip` to install the python packages: 
````
pip install pybullet
pip install attrdict
pip install collections
pip install joblib
pip install gc
````
Please look for the details of our simulation configurations in the directory `simulator`.

<!-- ## Simulation environment
The simulation environment will be available soon. --> 

## Training
``CUDA_VISIBLE_DEVICES=0,1 python train.py --tanh --grid --dataset_root path_to_dataset``

## Test
The Pretrained model is [here](https://drive.google.com/file/d/1Z8xUQmrzufVz7q-3hs9ZVVjjFCTPnxxB/view?usp=sharing).
````
CUDA_VISIBLE_DEVICES=0,1 python test.py --tanh --grid --dataset_root path_to_dataset --resume pretrained_model/checkpoint_440.pth.tar
````

Then it will generate the predicted grasps saved in `.npz` files in `pretrained_model/test/epoch440/view0`. The file `pretrained_model/test/epoch440/nms_poses_view0.txt` contains the predicted grasps after nms.

## Rule-based evaluation
You can use the following script to abtain the success rate and coverage rate.

````
CUDA_VISIBLE_DEVICES=0 python topk_percent_coverage_precision.py -pd pretrained_model/test/epoch440/view0 -gd path_to_gt_annotations
````

## Simulation-based evaluation
To test the predicted grasps in simulation environment:
````
cd simulator
python -m simulateTest.simulatorTestDemo -t pretrained_model/test/epoch440/nms_poses_view0.txt
````

## Citation
````
@article{wu2020grasp,
  title={Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps},
  author={Wu, Chaozheng and Chen, Jian and Cao, Qiaoyu and Zhang, Jianchi and Tai, Yunxin and Sun, Lin and Jia, Kui},
  journal={arXiv preprint arXiv:2009.12606},
  year={2020}
}
````

## Acknowledgement
The code of pointnet2 are borrowed from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
