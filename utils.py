import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import StratifiedKFold

def threshold_at_one(x):
    return x>75

def jpg_to_ndarray(path):
    image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def label_to_onehot(label, num_classes):
    a=label
    one_hot_a=np.zeros((num_classes,))
    one_hot_a[np.arange(num_classes)==a]=1
    #print(one_hot_a)
    return one_hot_a

def save_first_stage(tensor_dic, path):
    # tensor in shape: B, C, W, H
    B,C,H,W=tensor_dic['image'].shape
    for i in range(B):
        pass

def skfold(dataset):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
    for fold,(train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print("Train",train_ids, "Val", val_ids)
 

if __name__ =='__main__':
    pass
