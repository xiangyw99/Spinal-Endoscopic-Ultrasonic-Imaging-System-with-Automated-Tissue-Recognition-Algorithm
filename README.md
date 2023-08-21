# Spinal-Endoscopic-Ultrasonic-Imaging-System-with-Automated-Tissue-Recognition-Algorithm
Official PyTorch implementation for paper: ***Spinal Endoscopic Ultrasonic Imaging System with Automated Tissue Recognition Algorithm: Development and Optimization***

Follow the steps below to reproduce our result.
## Step 1: Setup
Run the following commands to create a conda environment. Make sure you are running on a linux system with one GPU.
```
conda create --name bmu --file requirements.txt
conda activate bmu
```
## Step 2: Dataset and Checkpoints
1. Download the dataset and the Checkpoints ([BaiduDisk](https://pan.baidu.com/s/1_DyzJzcJ7ASiZIz0n1WvlQ?pwd=h4r8), Password: h4r8).
2. Unzip `datasets.zip` and put the WHOLE dataset/checkpoints folder into your working directory.
3. Create an empty folder `logs/inference/`
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
│    ├────spine_image_0529_strict_split/
├────logs/
│    └────inference/
├────config.py
├────data_preprocessing.py
├────inference.py
├────train.py
├────trainer.py
└────utils.py
```
3. Go into `config.py`, change `imagesTr` and `model_path` to the path of dataset/checkpoints on your computer.  

## Step 3: Inference
Using the checkpoints we provided to reproduce our result. Run the command:
```
python inference.py --stage 2 --num_classes 6 --model densenet121 --pretrained True
```
It will automatically print the confusion matrix, metrics, and AUC score, etc.  
If you want to reproduce the result in the first layer or the second layer, run the command:
```
python inference.py --stage 0 --num_classes 3 --model densenet121 --pretrained True
python inference.py --stage 1 --num_classes 3 --model densenet121 --pretrained True
python inference.py --stage 1 --num_classes 2 --model densenet121 --pretrained True
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
python inference.py --stage 2 --num_classes 6 --model densenet121 --pretrained True
```
## Contact to us
If you have any problem to this repository, please send an issue or e-mail us: xiangyw99@outlook.com.
