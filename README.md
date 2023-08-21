# Spinal-Endoscopic-Ultrasonic-Imaging-System-with-Automated-Tissue-Recognition-Algorithm
Official PyTorch implementation for paper: **Spinal Endoscopic Ultrasonic Imaging System with Automated Tissue Recognition Algorithm: Development and Optimization**

Follow the steps below to reproduce our result.
## Step 1: Setup
Run the following commands to create a conda environment. Make sure you are running on a linux system with one GPU.
```
conda create -n bmu python=3.10
conda activate bmu
pip install -r requirements.txt
```
## Step 2: Preparing Dataset and Checkpoints
Prepare the dataset (BaiduDisk) and the Checkpoints (BaiduDisk). Put the WHOLE dataset/checkpoints folder into your working directory.  
Make sure your folder looks like:
```
WorkingDirection/
├────checkpoints/
│    ├────stage_0/
│    │    └────classes_3/
│    └────stage_1/
│    │    ├────classes_2/
│    │    └────classes_3/
├────datasets/
│    ├────spine_image_5_fold/
│    ├────spine_image_5_fold_stage2_class2/
│    └────spine_image_5_fold_stage2_class3/
├────logs/
│    └────inference/
├────config.py
├────data_preprocessing_60.py
├────inference.py
├────train.py
├────trainer.py
└────utils.py
```
## Step 3: Inference
Using the checkpoints we provided to reproduce our result. Run the command:
```
python inference.py --gpu 0 --stage 2 
```
It will automatically print the confusion matrix, metrics, and AUC score, etc.  
If you want to reproduce the result in the first layer or the second layer, run the command:
```
python inference.py --gpu 0 --stage 0 --num_classes 3
python inference.py --gpu 0 --stage 1 --num_classes 3
python inference.py --gpu 0 --stage 1 --num_classes 2
```

## Step 4 (Optional): Train
If you want to train on your own to reproduce our result, run the command:
```
python train.py --gpu 0 --stage 0 --num_classes 3  --model densenet121 --pretrained True --experiment densenet121
python train.py --gpu 0 --stage 1 --num_classes 3  --model densenet121 --pretrained True --experiment densenet121
python train.py --gpu 0 --stage 1 --num_classes 2  --model densenet121 --pretrained True --experiment densenet121
```
Then run the inference:
```
python inference.py --gpu 0 --stage 2 
```
