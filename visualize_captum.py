import os
import numpy as np
import argparse
import torchvision                                                                                                                                               
from torchvision import models                                                                                                                                   
from torchvision import transforms                                                                                                                               
                                                                                                                                                                 
from captum.attr import IntegratedGradients                                                                                                                      
from captum.attr import GradientShap                                                                                                                             
from captum.attr import LRP                                                                                                                                      
from captum.attr import Occlusion                                                                                                                                
from captum.attr import NoiseTunnel                                                                                                                              
from captum.attr import visualization as viz                                                                                                                     
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule                                                                               
                                                                                                                                                                 
import torch                                                                                                                                                     
import torch.nn as nn                                                                                                                                            
import torch.nn.functional as F                                                                                                                                  
import torch.backends.cudnn as cudnn                                                                                                                             
from basic_code import load, util, OptimalThresholdSensitivitySpecificity
from basic_code import networks_captum as networks                                                                                                    
#from basic_code import networks_captum_regression as networks
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve                                                                                           
from PIL import Image
import json
from matplotlib.colors import LinearSegmentedColormap

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cate2label = {0: 'healthy', 1: 'dementia', 'healthy': 0, 'dementia': 1}

task = 'PupilCalib'
#task = 'CookieTheft' 
#task = 'Reading'                                                                                                                                                
itt = 1                                                                                                                                                          
cross_fold = str(1)                                                                                                                                              
load_file_path = 'data/txt/iter_'+str(itt)+'/'                                                            
load_img_path = 'data/face/train/'
save_img_path = 'data/visualization/' + task + '/iter_'+str(itt)+'/'+cross_fold+'/'
model_test_path = task + '/iter_'+str(itt)+'/'
at_type = 'self_relation-attention'

_structure = networks.resnet18_at(at_type = at_type)
_parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar' 
model = load.model_parameters(_structure, _parameterDir) 
model.module.pred_fc1 = nn.Linear(512, 2).cuda()
model.module.pred_fc2 = nn.Linear(1024, 2).cuda()
model_file = os.listdir(model_test_path+cross_fold)[0]
_parameterDir = model_test_path+cross_fold+'/'+model_file
print(cross_fold, _parameterDir)
model = load.model_parameters(_structure, _parameterDir)
model.eval()

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )


integrated_gradients = IntegratedGradients(model)
default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                 [(0, '#ffffff'),
                  (0.25, '#000000'),
                  (1, '#000000')], N=256)

file_name = open(load_file_path+'Test'+str(itt)+'-'+task+'.txt').read().split('\n')[:-1]
for fff in file_name:
    
    img_path = load_img_path + fff.split(' ')[0] + '/'

    if not os.path.exists(save_img_path + fff.split(' ')[0].split('/')[1]):
        os.makedirs(save_img_path + fff.split(' ')[0].split('/')[1])

    for ele in os.listdir(img_path):

        if ele in os.listdir(save_img_path + fff.split(' ')[0].split('/')[1]):
            continue
        img = Image.open(img_path+ele)
        transformed_img = transform(img)
        input_img = transform_normalize(transformed_img)
        input_img = input_img.unsqueeze(0).to(DEVICE)
        

        output = model(input_img)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = cate2label[int(str(pred_label_idx.item()))]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(input_img, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        vis_img, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
              ["original_image", "heat_map"],
              ["all", "positive"],
              cmap=default_cmap,
              show_colorbar=True)

        vis_img = vis_img.savefig(save_img_path + fff.split(' ')[0].split('/')[1] + '/' + ele)
