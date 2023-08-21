import os
import torch


def train_one_fold(fold, args, model, logger, train_loader, val_loader, optimizer, loss_fn, scheduler):
    # check if model path exists
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    path='./checkpoints/stage_{}/classes_{}/fold_{}'.format(args.stage, args.num_classes, fold)
    if not os.path.exists(path):
        os.mkdir(path)

    best_val=0.
    for epoch in range(args.epochs):
        epoch_loss=0.
        model.train()
        logger.info("Epoch: "+str(epoch)) 
        for i,batch in enumerate(train_loader):
            
            image=batch['image'].cuda()
            image=torch.cat([image,image,image], dim=1).cuda()

            # label=torch.from_numpy(label_to_onehot(batch['label'])).cuda()
            if args.stage==0:
                label=batch['label_stage1'].cuda()
            elif args.stage==1:
                label=batch['label_stage2'].cuda()
            elif args.stage==3:
                label=batch['label'].cuda()
            
            optimizer.zero_grad()

            output = model(image)
            #label=torch.from_numpy(label_to_onehot(label.cpu().numpy(), args.num_classes)).cuda()
            loss=loss_fn(output, label)

            loss.backward()
            
            epoch_loss+=loss.item()

            optimizer.step()
            
        scheduler.step()
    
        logger.info(f"loss: {epoch_loss/(i+1):.7f}, lr: {scheduler._last_lr[0]:.7f}")
        

        # torch.save(model, path+'/checkpoint_latest_{}_pretrained_{}.pt'.format(args.model, str(args.pretrained)))

        model.eval()
        predict_num=0
        target_num=0
        acc_num=0
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
                outputs=model(image)
                predicted = torch.max(outputs.data,1)[1]
            
                pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1).long(), 1.)
                predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
                tar_mask = torch.zeros(outputs.size()).scatter_(1, label.data.cpu().view(-1, 1).long(), 1.)
                target_num += tar_mask.sum(0)  # 得到数据中每类的数量
                acc_mask = pre_mask * tar_mask 
                acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            accuracy = 100. * acc_num.sum(0) / target_num.sum(0)

            
            #logger.info(f"Val Dice Score: {metric:.3f}")
            #dice_metric.reset()

            if accuracy>best_val:
                logger.info(f"YEAYYYYYY, saving a better model, acc: {accuracy:.3f}")
                best_val=accuracy
                
                torch.save(model, path+'/checkpoint_best_{}_pretrained_{}.pt'.format(args.model, str(args.pretrained)))
                logger.info('Test Acc {}'.format(accuracy))
                logger.info('Recal {}'.format(recall))
                logger.info('Precision {}'.format(precision))
                logger.info('F1-score {}'.format(F1))
            elif accuracy==100.:
                break
            else:
                logger.info(f'Best acc: {best_val:.3f}, Current acc: {accuracy:.3f}')
    # val loop
    logger.info(f"Stage {args.stage}, fold: {fold}, best val acc: {best_val}")
    return best_val
