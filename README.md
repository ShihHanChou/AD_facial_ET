# AD_facial_ET
Official code for Multimodal Classification of Alzheimer’s Disease by Combining Facial and Eye-Tracking Data.

### dependencies
```
# create the environment for the project
conda create -n facial python=3.9
conda activate facial

# install cv2
pip install opencv-python
```
install [pytorch](https://pytorch.org/get-started/locally/)

### download pretrain models and published dataset
Download pretrained model ***Resnet18_FER+_pytorch.pth.tar***. [Baidu](https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg) or [OneDrive](https://1drv.ms/u/s!AhGc2vUv7IQtl1Pt7FhPXr_Kofd5?e=3MvPFX) . Please put the model in the directory: ***"./pretrain_model/"***. 

For data preparation, please follow [EMOTION_FAN repo](https://github.com/Open-Debin/Emotion-FAN/tree/master) to extract frames from videos and extract faces from frames.

### running experiments
```
# Training with relation-attention
CUDA_VISIBLE_DEVICES=0 python facial_traintest.py --type 1
