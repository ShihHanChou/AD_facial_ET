import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from basic_code import load, util, networks
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    parser = argparse.ArgumentParser(description='PyTorch Frame Attention Network Training')
    parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',
                        help= '0 is self-attention; 1 is self + relation-attention')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=4e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()
   
    task = 'PupilCalib' 
    #task = 'CookieTheft'
    #task = 'Reading'
    save_path = './model_10_Harshinee_task/'+task
    for itt in range(2,11):  
        ''' Load data '''
        for t_fold in range(10):

            best_acc = 0
            at_type = ['self-attention', 'self_relation-attention'][args.at_type]
            logger = util.Logger('./log/','fan_afew')
            logger.print('The attention method is {:}, learning rate: {:}'.format(at_type, args.lr))

            cross_fold = str(t_fold)
            #root_train = './data/face/train_afew_whole_video_concat'
            root_train = './data/face/train_afew'
            #list_train = './data/txt/10_cross_Harshinee_v2/Train'+cross_fold+'.txt'
            list_train = './data/txt/10_cross_Harshinee_task/iter_'+str(itt)+'/Train'+cross_fold+'-'+task+'.txt'
            batchsize_train= 48
            #root_eval = './data/face/train_afew_whole_video_concat'
            root_eval = './data/face/train_afew'
            #list_eval = './data/txt/10_cross_Harshinee_v2/Val'+cross_fold+'.txt'
            list_eval = './data/txt/10_cross_Harshinee_task/iter_'+str(itt)+'/Val'+cross_fold+'-'+task+'.txt'
            batchsize_eval= 65
            train_loader, val_loader = load.afew_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval)
            ''' Load model '''
            _structure = networks.resnet18_at(at_type=at_type)
            _parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
            model = load.model_parameters(_structure, _parameterDir)
            model.module.pred_fc1 = nn.Linear(512, 2).cuda()
            model.module.pred_fc2 = nn.Linear(1024, 2).cuda()
            #_parameterDir = './model_relation_correct/original/self_relation-attention_1_74.2424'
            #model = load.model_parameters(_structure, _parameterDir)
            ''' Loss & Optimizer '''
            #train_para = list(model.module.pred_fc1.parameters()) + list(model.module.pred_fc2.parameters())
            #optimizer = torch.optim.SGD(train_para, args.lr, momentum=0.9, weight_decay=1e-4)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            cudnn.benchmark = True
            ''' Train & Eval '''
            if args.evaluate == True:
                logger.print('args.evaluate: {:}', args.evaluate)        
                val(val_loader, model, at_type, logger)
                return
            logger.print('frame attention network (fan) afew dataset, learning rate: {:}'.format(args.lr))
            
            for epoch in range(args.epochs):
                train(train_loader, model, optimizer, epoch, logger)
                acc_epoch = val(val_loader, model, at_type, logger)
                is_best = acc_epoch > best_acc
                if is_best:
                    logger.print('better model!')
                    best_acc = max(acc_epoch, best_acc)
                    util.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'accuracy': acc_epoch,
                        'cross_fold': cross_fold,
                        'iter': str(itt),
                        'save_path': save_path,
                    }, at_type=at_type)
                    
                lr_scheduler.step()
                logger.print("epoch: {:} learning rate:{:}".format(epoch+1, optimizer.param_groups[0]['lr']))
        
def train(train_loader, model, optimizer, epoch, logger):
    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []
    index_vector = []

    model.train()
    for i, (input_first, input_second, input_third, target_first, index) in enumerate(train_loader):
        target_var = target_first.to(DEVICE)
        input_var = torch.stack([input_first, input_second , input_third], dim=4).to(DEVICE)
        # compute output
        ''' model & full_model'''
        pred_score = model(input_var)
        loss = F.cross_entropy(pred_score, target_var)
        loss = loss.sum()
        #
        output_store_fc.append(pred_score)
        target_store.append(target_var)
        index_vector.append(index)
        # measure accuracy and record loss
        acc_iter = util.accuracy(pred_score.data, target_var, topk=(1,))
        losses.update(loss.item(), input_var.size(0))
        topframe.update(acc_iter[0], input_var.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            logger.print('Epoch: [{:3d}][{:3d}/{:3d}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, topframe=topframe))

    index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)

    index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
    output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
    target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [380,21570] * [21570, 7] = [380,7]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]

    acc_video = util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu(), topk=(1,))
    topVideo.update(acc_video[0], i + 1)
    logger.print(' *Acc@Video {topVideo.avg:.3f}   *Acc@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))

def val(val_loader, model, at_type, logger):
    topVideo = util.AverageMeter()
    # switch to evaluate mode
    model.eval()
    output_store_fc = []
    output_alpha    = []
    target_store = []
    index_vector = []
    with torch.no_grad():
        for i, (input_var, target, index) in enumerate(val_loader):
            # compute output
            target = target.to(DEVICE)
            input_var = input_var.to(DEVICE)
            ''' model & full_model'''
            f, alphas = model(input_var, phrase = 'eval')

            output_store_fc.append(f)
            output_alpha.append(alphas)
            target_store.append(target)
            index_vector.append(index)

        index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)

        index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE).float()  # [21570]  --->  [380, 21570]
        output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
        output_alpha    = torch.cat(output_alpha, dim=0)     # [256,1] ... [256,1]  --->  [21570, 1]
        target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
        ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
        weight_sourcefc = output_store_fc.mul(output_alpha)   #[21570,512] * [21570,1] --->[21570,512]
        sum_alpha = index_matrix.mm(output_alpha) # [380,21570] * [21570,1] -> [380,1]
        weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha)
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]
        if at_type == 'self-attention':
            pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
        if at_type == 'self_relation-attention':
            pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')
        acc_video = util.accuracy(pred_score.cpu(), target_vector.cpu(), topk=(1,))
        output_results = np.array(torch.argmax(pred_score.cpu(), dim=1))
        np.save('output_results.npy', output_results)
        topVideo.update(acc_video[0], i + 1)
        logger.print(' *Acc@Video {topVideo.avg:.4f} '.format(topVideo=topVideo))
        return topVideo.avg
if __name__ == '__main__':
    main()
