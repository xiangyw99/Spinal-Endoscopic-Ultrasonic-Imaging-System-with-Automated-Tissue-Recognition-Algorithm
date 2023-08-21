# Spinal-Endoscopic-Ultrasonic-Imaging-System-with-Automated-Tissue-Recognition-Algorithm
Official PyTorch implementation for paper: Spinal Endoscopic Ultrasonic Imaging System with Automated Tissue Recognition Algorithm: Development and Optimization

Follow the steps below to reproduce our result.
## Step 1: Setup
Run the following commands to create a conda environment. Make sure you are running on a linux system with one GPU.
```
conda create -n bmu python=3.10
conda activate bmu
pip install -r requirements.txt
```
## Step 2: Prepare Dataset and Model
Prepare the dataset (Download) and the model weights (Download) in the folder. Make sure your folder looks like:

## Step 3: Inference
Using the trained weights to reproduce our result. Run the command:
```
python inference.py --stage 2 --gpu 0 --model densenet121
```

## Step 4 (Optional): Train
If you want to train on your own, run the command:
```
python train.py --gpu 0 --stage 0 --num_classes 3 --imagesTr $YOUR DATASET ROOT$  --model densenet121 --pretrained True
python train.py --gpu 0 --stage 1 --num_classes 3 --imagesTr $YOUR DATASET ROOT$  --model densenet121 --pretrained True
python train.py --gpu 0 --stage 1 --num_classes 2 --imagesTr $YOUR DATASET ROOT$  --model densenet121 --pretrained True
```
Then run the inference root:
```
python inference.py --stage 2 --gpu 0 --model densenet121
```
