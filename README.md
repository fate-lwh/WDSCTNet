Title: Single-Branch CNN with Generative Denoising Weighted Distillation for Real-Time Semantic Segmentation in Complex Climate Driving Scenarios
=

----

Datasets
----
We trained by mixing the [Cityscapes](https://www.cityscapes-dataset.com/) and [ACDC](https://acdc.vision.ee.ethz.ch/) datasets.




Environment Installation
---
```
conda create -n SCTNet python=3.8
conda activate SCTNet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install timm
pip install matplotlib
pip install prettytable
pip install einops
```


Training
---
```
# 2 gpus, batch_per_gpu=8 , btachsize =16
bash tools/dist_train.sh configs\WDSCTNet\cityscapes\wdsctnet-b_seg75_8x2_160k_cityscapes.py 2
```
