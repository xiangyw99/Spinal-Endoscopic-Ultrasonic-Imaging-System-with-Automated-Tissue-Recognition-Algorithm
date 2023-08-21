import os
import torch
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data_preprocessing import ABUSsetNew
import logging
from config import CFG
from monai.utils import set_determinism
from sklearn.metrics import roc_auc_score

classes=['SC','NP','AT','Bone','AF','NR']
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

    parser.add_argument('-o', '--output_dir', type=str, default=CFG.output_dir, required=False, help='Directory to save logs')

    # training
    parser.add_argument('--lr', type=float, default=CFG.lr, help='learning rate')
    parser.add_argument('--epochs', type=int, default=CFG.epochs, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size, help='batch size')
    parser.add_argument('--imagesTr', type=str, default=CFG.imagesTr, help='dataset root')

    parser.add_argument('--nw', type=int, default=CFG.nw, help='number of workers')

    parser.add_argument('--model_path', type=str, default=CFG.model_path, help='checkpoint path, only for validation')
    parser.add_argument('--model', type=str, default=CFG.model, help='densenet121, resnet34 ...')
    parser.add_argument('--pretrained', type=bool, default=CFG.pretrained, help='Use pretrained weights or not')
    parser.add_argument('--num_classes', type=int, default=CFG.num_class, help='num_classes')
    parser.add_argument('--k_folds', type=int, default=CFG.k_folds, help='k-fold cross validation')

    parser.add_argument('--scheduler', type=str, default=CFG.scheduler, help='lr scheduler')
    parser.add_argument('--experiment', type=str, default=CFG.experiment, help='experiment name')

    parser.add_argument('--stage', type=int, required=True, help='0 for training the first layer, 1 for training the second layer, 2 for validation')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    return parser


def create_logger(args):
    """
    :param logger_file_name:
    :return:
    """
    path=args.output_dir+'/inference'
    if not os.path.exists(path):
        os.mkdir(path)
    loca=time.strftime('%Y-%m-%d-%H-%M-%S')
    logger_file_name=path+"/"+str(loca) + ".txt"
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
                                                             eta_min=CFG.min_lr, T_mult=2)
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

logger.info("Begin to inference")

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



confuse_matrix=torch.zeros((args.num_classes, args.num_classes))
sample_statistic_matrix=torch.zeros((args.num_classes, args.num_classes))
metrice_matrix=torch.zeros((args.num_classes, 4))  # tp,fp,tn,fn

kfold_val_acc=[]
total_time=[]
label_queue=[]
pred_queue=[]
logits_total=torch.zeros([367, 6])
response_speed=np.zeros([367,2])

counter=0
for fold in range(5):
    # Print
    dataset=ABUSsetNew(args, fold=fold, mode='val')
    
    logger.info('--------------------------------')
    logger.info(f'FOLD {fold}')
    val_loader=DataLoader(dataset, batch_size=1,  num_workers=4)
# model

    if args.stage==2:
        path1=args.model_path+f'/stage_0/classes_3/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        path2=args.model_path+f'/stage_1/classes_2/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        path3=args.model_path+f'/stage_1/classes_3/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        model1=torch.load(path1)
        model2=torch.load(path2)
        model3=torch.load(path3)
        model1.cuda().eval()
        model2.cuda().eval()
        model3.cuda().eval()

        correct=0
        total=0
        
        with torch.no_grad():
            
            for i,batch in enumerate(val_loader):
                
                image=batch['image'].cuda()
                image=torch.cat([image,image,image], dim=1)

                label=batch['label'].cpu().numpy()
                a=time.time()
                outputs1=model1(image)
                cls=torch.argmax(outputs1)
                
                if cls==0:
                    final=model3(image)
                    prob=torch.exp(final)/(torch.sum(torch.exp(final)))
                    logits_total[counter,0:3]=prob
                    
                    logits=final.cpu().numpy()
                    final=torch.argmax(final)
                    final=final.item()
                elif cls==1:
                    outputs2=model2(image)
                    prob=torch.exp(outputs2)/(torch.sum(torch.exp(outputs2)))
                    logits_total[counter,3:5]=prob
                    logits=outputs2.cpu().numpy()
                    outputs2=torch.argmax(outputs2)
                    if outputs2==0:
                        final=3
                    elif outputs2==1:
                        final=4
                elif cls==2:
                    prob=torch.exp(outputs1)/(torch.sum(torch.exp(outputs1)))
                    
                    logits_total[counter, 0:3]=prob[0,0]/3
                    logits_total[counter,3:5]=prob[0,1]/2
                    logits_total[counter,5]=1-prob[0,0]/3-prob[0,1]/2
                    logits=outputs1.cpu().numpy()
                    final=5
                if final==label:
                    correct+=1
                b=time.time()
                if batch['name'][0]=='gutou_AScan20230529_185452.npy':
                    
                    img=image.cpu().numpy()
                    import pdb
                    pdb.set_trace()
                response_speed[counter,0]=label.item()
                response_speed[counter,1]=b-a
                total+=1
                counter+=1
                confuse_matrix[label, final]+=1
                total_time.append(b-a)
                label_queue.append(label.item())
                pred_queue.append(final)
                #print(f'Total: {total}, corr:{correct}')
            logger.info(f"Fold {fold} final acc: {100*correct/total:.2f}, ")
            kfold_val_acc.append(100*correct/total)
    elif args.stage==0:
        path=args.model_path+f'/stage_{args.stage}/classes_3/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        model=torch.load(path)
        model.cuda().eval()
        correct=0
        total=0

        with torch.no_grad():
            for i,batch in enumerate(val_loader):
                image=batch['image'].cuda()
                image=torch.cat([image,image,image], dim=1)
                if args.stage==0:
                    label=batch['label_stage1'].cuda()
                elif args.stage==1:
                    label=batch['label_stage2'].cuda()
                elif args.stage==3:
                    label=batch['label'].cuda()
                label=label.cpu().numpy()
            
                outputs1=model(image)
                cls=torch.argmax(outputs1)
                
                
                if cls==label:
                    correct+=1
                total+=1
                confuse_matrix[label, cls]+=1
            
                #print(f'Total: {total}, corr:{correct}')
            logger.info(f"Fold {fold} final acc: {100*correct/total:.2f}")
            kfold_val_acc.append(100*correct/total)
    elif args.stage==1:
        path=args.model_path+f'/stage_{args.stage}/classes_{args.num_classes}/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        
        model=torch.load(path)
        model.cuda().eval()
        correct=0
        total=0

        with torch.no_grad():
            for i,batch in enumerate(val_loader):
                image=batch['image'].cuda()
                image=torch.cat([image,image,image], dim=1)
                if args.stage==0:
                    label=batch['label_stage1'].cuda()
                elif args.stage==1:
                    label=batch['label_stage2'].cuda()
                elif args.stage==3:
                    label=batch['label'].cuda()
                label=label.cpu().numpy()
            
                outputs1=model(image)
                cls=torch.argmax(outputs1)
                if cls==label:
                    correct+=1
                total+=1
                confuse_matrix[label, cls]+=1
            
                #print(f'Total: {total}, corr:{correct}')
            logger.info(f"Fold {fold} final acc: {100*correct/total:.2f}")
            kfold_val_acc.append(100*correct/total)
    elif args.stage==3:
        
        path=args.model_path+f'/stage_3/classes_6/fold_{fold}/checkpoint_best_{args.model}_pretrained_{args.pretrained}.pt'
        
        model=torch.load(path)
        
        model.cuda().eval()

        correct=0
        total=0
        
        with torch.no_grad():
            
            for i,batch in enumerate(val_loader):
                a=time.time()
                image=batch['image'].cuda()
                image=torch.cat([image,image,image], dim=1)

                label=batch['label'].cpu().numpy()
                outputs=model(image)
                prob=torch.exp(outputs)/(torch.sum(torch.exp(outputs)))
                final=torch.argmax(outputs)
                logits_total[counter,:]=prob
                
                if final==label:
                    correct+=1
                b=time.time()
                total+=1
                response_speed[counter,0]=label.item()
                response_speed[counter,1]=b-a
                counter+=1
                confuse_matrix[label, final]+=1
                sample_statistic_matrix[label,label]+=1
                total_time.append(b-a)
                label_queue.append(label.item())
                pred_queue.append(final.cpu().numpy())
                
                #print(f'Total: {total}, corr:{correct}')
            logger.info(f"Fold {fold} final acc: {100*correct/total:.2f}, ")
            kfold_val_acc.append(100*correct/total)


print("-----------------------------")
logger.info(f"{fold+1}-fold cross validation acc: {sum(kfold_val_acc)/(fold+1):.2f}%")
print("-----------------------------")
print("Confusion matrix")
print(confuse_matrix)
#logits_total=np.round(logits_total.numpy(),4)
if args.stage==2:
    logits_total=logits_total.numpy()
    print("-----------------------------")
    print("Prediction in probabilities")
    print(logits_total)

    for i in range(367):
        logits_total[i,0]=1.0000-np.sum(logits_total[i,1:6])

    #for i in range(367):
    #   assert np.sum(logits_total[i,:])==1, f'got logit in line: {i} sum not equal to 1, but equals to: {np.sum(logits_total[i,:])}got logits: {logits_total[i,:]}'

    auc=roc_auc_score(label_queue, logits_total, multi_class='ovr')
    print("-----------------------------")
    print("ROC score (multiclasses)")
    print(auc)

    print("-----------------------------")
    print("ROC score (single class)")
    for i in range(6):

        y_pred=logits_total[np.arange(len(label_queue)),i]
    
        y_score=logits_total[:,i]
        label_queue=np.array(label_queue)
        binary_label=np.zeros_like(label_queue)
        binary_label[label_queue==i]=1
        binary_label[label_queue!=i]=0
        auc=roc_auc_score(binary_label, y_score)
        print("class "+classes[i]+":",auc)