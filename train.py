import os
import torch

import time
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from torch.optim import lr_scheduler
from data_preprocessing_60 import ABUSsetNew
#from data_preprocessing_two_stage import ABUSset
import timm
import logging
from config import CFG
from utils import label_to_onehot
from monai.utils import set_determinism
from trainer import train_one_fold_for_six, train_one_fold
from sklearn.model_selection import KFold, StratifiedKFold

def set_seed(seed = 2023):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    set_determinism(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

set_seed(2023)


def parse_training_args(parser):

    parser.add_argument('-o', '--output_dir', type=str, default=CFG.output_dir, required=False, help='Directory to save checkpoints')

    # training
    parser.add_argument('--lr', type=float, default=CFG.lr, help='000')
    parser.add_argument('--epochs', type=int, default=CFG.epochs, help='000')
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size, help='000')
    parser.add_argument('--patch_size', type=tuple, default=CFG.patch_size, help='000')
    parser.add_argument('--imagesTr', type=str, default=CFG.imagesTr, help='000')
    parser.add_argument('--labelsTr', type=str, default=CFG.labelsTr, help='000')
    parser.add_argument('--imagesTs', type=str, default=CFG.imagesTs, help='000')
    parser.add_argument('--labelsTs', type=str, default=CFG.labelsTs, help='000')
    parser.add_argument('--nw', type=int, default=CFG.nw, help='000')

    parser.add_argument('--c', type=bool, default=CFG.c, help='000')
    parser.add_argument('--model_path', type=str, default=CFG.model_path, help='000')
    parser.add_argument('--model', type=str, default=CFG.model, help='unet_3D, attention_unet, voxresnet, vnet, nnUNet')
    parser.add_argument('--pretrained', type=bool, default=CFG.pretrained, help='unet_3D, attention_unet, voxresnet, vnet, nnUNet')
    parser.add_argument('--in_class', type=int, default=CFG.in_class, help='000')
    parser.add_argument('--num_classes', type=int, default=CFG.num_class, help='000')
    parser.add_argument('--k_folds', type=int, default=CFG.k_folds, help='000')

    parser.add_argument('--scheduler', type=str, default=CFG.scheduler, help='000')
    parser.add_argument('--label', type=str, default='label', help='000')
    parser.add_argument('--experiment', type=str, default=CFG.experiment, help='000')

    parser.add_argument('--stage', type=int, required=True, help='000')
    parser.add_argument('--gpu', type=str, required='0', help='000')
    return parser


def create_logger(args):
    """
    :param logger_file_name:
    :return:
    """
    path=args.output_dir
    if not os.path.exists(path):
        os.mkdir(path)
    loca=time.strftime('%Y-%m-%d-%H-%M-%S')
    logger_file_name=args.output_dir+"/"+args.model+str(loca) + ".txt"
    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def fetch_scheduler(args, optimizer, steps_per_epoch):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.epochs*steps_per_epoch/CFG.n_accumulate, 
                                                   eta_min=CFG.min_lr)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif args.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    elif args.scheduler == None:
        return None
    else:
        raise RuntimeError("666 what is it")
    return scheduler

parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
parser = parse_training_args(parser)
args, _ = parser.parse_known_args()
args = parser.parse_args()

logger = create_logger(args)

logger.info("Begin to train the model: {} , Pretrained: {}".format(args.model,  str(args.pretrained)))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
devicess = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset and dataloader
#dataset=ABUSsetNew(args.imagesTr, mode='train')
# val_set=ABUSset(args.imagesTs, mode='train', postfix='.jpg')   ################### important!!! this line the 'mode'==train for k-fold cross-validation

kfold_val_acc=[]
#kfold=KFold(shuffle=True, random_state=2023)
#for fold,(train_ids, val_ids) in enumerate(kfold.split(dataset)):
    # Print
for fold in range(5):  
    logger.info(f'FOLD {fold}')
    logger.info('--------------------------------')

    trainset=ABUSsetNew(args, fold=fold, mode='train')
    valset=ABUSsetNew(args, fold=fold, mode='val')

    #train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    #train_loader=DataLoader(dataset, batch_size=args.batch_size,  pin_memory=True, num_workers=4, sampler=train_subsampler)
    #val_loader=DataLoader(dataset, batch_size=args.batch_size,  num_workers=4, sampler=val_subsampler)

    train_loader=DataLoader(trainset, shuffle=True,batch_size=args.batch_size,  pin_memory=True, num_workers=4)
    val_loader=DataLoader(valset, shuffle=True,batch_size=args.batch_size,  num_workers=4)

    # model
    if args.c:
        model=torch.load(args.model_path)
        logger.info("Continue to train from "+args.model_path)
    else:
        model=timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes).cuda()
        logger.info("Training from scratch")

    # loss, optimizer, scheduler
    if args.stage==0:
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1/150,1/100,1/50]).cuda())
    else: loss_fn=nn.CrossEntropyLoss()

    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler=fetch_scheduler(args, optimizer, steps_per_epoch=400//args.batch_size)
    val_acc=train_one_fold(fold=fold, args=args, model=model, logger=logger, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler)
    kfold_val_acc.append(val_acc)
logger.info(f"Stage: {args.stage}, {args.k_folds}-fold average val acc {sum(kfold_val_acc)/args.k_folds}(model {args.model}, pretrained {str(args.pretrained)}, experiment {args.experiment}, num_classes {args.num_classes})")
