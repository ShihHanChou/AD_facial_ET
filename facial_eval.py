import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from basic_code import load, util, networks, OptimalThresholdSensitivitySpecificity
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


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
    best_acc = 0
    at_type = ['self-attention', 'self_relation-attention'][args.at_type]
    logger = util.Logger('./log/','facial')
    logger.print('The attention method is {:}, learning rate: {:}'.format(at_type, args.lr))
  
    task = 'PupilCalib' 
    #task = 'CookieTheft' 
    #task = 'Reading'
    AUC_score_all, sensitivity_all, specificity_all, accuracy_all = [], [], [], []
    for itt in range(1,11): 
        AUC_score_iter, sensitivity_iter, specificity_iter, accuracy_iter = [], [], [], []
        for t_fold in range(10):

            ''' Load data '''
            cross_fold = str(t_fold)
            root_train = './data/face/train'
            list_train = './data/txt/iter_'+str(itt)+'/Train'+cross_fold+'-'+task+'.txt'
            batchsize_train= 48
            root_eval = './data/face/train'
            list_eval = './data/txt/iter_'+str(itt)+'/Test'+cross_fold+'-'+task+'.txt'
            batchsize_eval= 65
            train_loader, val_loader = load.afew_faces_fan(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval)
            ''' Load model '''
            _structure = networks.resnet18_at(at_type=at_type)
            _parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
            model = load.model_parameters(_structure, _parameterDir)
            model.module.pred_fc1 = nn.Linear(512, 2).cuda()
            model.module.pred_fc2 = nn.Linear(1024, 2).cuda()
            model_test_path = './'+task+'/iter_'+str(itt)+'/'
            model_file = os.listdir(model_test_path+cross_fold)[0]
            _parameterDir = model_test_path+cross_fold+'/'+model_file
            print(cross_fold, _parameterDir)
            model = load.model_parameters(_structure, _parameterDir)
            ''' Loss & Optimizer '''
            cudnn.benchmark = True
            ''' Train & Eval '''
            if args.evaluate == True:
                logger.print('args.evaluate: {:}', args.evaluate)        
                _, AUC_score, sensitivity, specificity, accuracy = val(val_loader, model, at_type, logger)
                AUC_score_iter.append(AUC_score)
                sensitivity_iter.append(sensitivity)
                specificity_iter.append(specificity)
                accuracy_iter.append(accuracy)
        AUC_score_all.append(np.mean(AUC_score_iter))
        sensitivity_all.append(np.mean(sensitivity_iter))
        specificity_all.append(np.mean(specificity_iter))
        accuracy_all.append(np.mean(accuracy_iter))
    print(AUC_score_all)
    print(sensitivity_all)
    print(specificity_all)
    print(accuracy_all)
    
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
        target_vector[target_vector<0] = 0
        if at_type == 'self-attention':
            pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
        if at_type == 'self_relation-attention':
            pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')
        acc_video = util.accuracy(pred_score.cpu(), target_vector.cpu(), topk=(1,))
        output_results = np.array(torch.argmax(pred_score.cpu(), dim=1))
        np.save('output_results.npy', output_results)
        topVideo.update(acc_video[0], i + 1)
        logger.print(' *Acc@Video {topVideo.avg:.4f} '.format(topVideo=topVideo))

        y_true_flipped = np.array(target_vector.cpu().numpy(), copy=True)
        y_true_flipped[target_vector.cpu().numpy() == 1] = 0
        y_true_flipped[target_vector.cpu().numpy() == 0] = 1

        pred_softmax = nn.Softmax(dim=1)
        pred_score_softmax = pred_softmax(pred_score)
        AUC_score = roc_auc_score(y_true_flipped, pred_score_softmax.cpu()[:,0])
        
        fpr, tpr, thresholds = roc_curve(target_vector.cpu(), pred_score_softmax.cpu()[:,0], pos_label=0)
        sensitivity, specificity, accuracy = OptimalThresholdSensitivitySpecificity.optimal_threshold_sensitivity_specificity(thresholds[1:], tpr[1:], fpr[1:], target_vector.cpu(), pred_score_softmax.cpu()[:,0])

        print('AUC:', AUC_score, 'Sensitivity:', sensitivity, 'Specificity:', specificity, 'Accuracy:', accuracy)

        return topVideo.avg, AUC_score, sensitivity, specificity, accuracy
if __name__ == '__main__':
    main()
