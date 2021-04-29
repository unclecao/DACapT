import os
import scipy.io
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import time
from capsule.capsule_layer import CapsuleLayer
#from roi_module import RoIPooling2D
#from new_roipool.roi_pooling.modules.roi_pool import _RoIPooling # another version of roipool
from new_roialign.roi_align.modules.roi_align import RoIAlignAvg # 

from utils import totensor, featureVisualize, squash
###########
# Deal with pytorch-0.3.1
# =============================================================================
# import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# =============================================================================
############

def append_params(params, module, prefix):
    m=0
    for child in module.children():
        if isinstance(child, CapsuleLayer): # For use routing
            for k,p in child._parameters.iteritems():
                name = prefix + '_prediction_' + k
                if name not in params:
                    params[name] = p
                else:
                    raise RuntimeError("Duplicated param name: %s" % (name))
            for i,j in enumerate(child.children()):
                y=0
                for d,q in enumerate(j.children()):
                    for k,p in q._parameters.iteritems():
                        if p is None: continue
                        if isinstance(q, nn.BatchNorm2d):
                            name = prefix + '_prediction'+'_bn_' + k + str(m)+ str(y)
                        else:
                            name = prefix + '_prediction'+'_' + k + str(m) + str(y)
                        if name not in params:
                            params[name] = p
                        else:
                            raise RuntimeError("Duplicated param name: %s" % (name))
                        y=y+1
                m = m+1
        else:
            if prefix == 'capPrimary':
                y=0
                for i,j in enumerate(child.children()):
                    for k,p in j._parameters.iteritems(): 
                        if p is None: continue
                        if isinstance(j, nn.BatchNorm2d):
                            name = prefix + '_bn_' + k + str(m)+ str(y)
                        else:
                            name = prefix + '_' + k + str(m) + str(y)
                        if name not in params:
                            params[name] = p
                        else:
                            raise RuntimeError("Duplicated param name: %s" % (name))
                        y=y+1
                m = m+1
            else:
                for k,p in child._parameters.iteritems():
                    if p is None: continue
        
                    if isinstance(child, nn.BatchNorm2d):
                        name = prefix + '_bn_' + k
                    else:
                        name = prefix + '_' + k
        
                    if name not in params:
                        params[name] = p
                    else:
                        raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class capMDNet(nn.Module):
    def __init__(self, roi_size, spatial_scale, num_input_primary_capsule_channel, num_output_primary_capsule_channel, num_primary_unit, 
                 primary_unit_size, num_predictions, output_unit_size, num_routing, model_path=None, K=1, network_status = 'Train', 
                 regrouping_type = None, group_attention =True, high_cap_conv = False, single_conv = None, noTM = False, 
                 penalty_attention =True, fc = False, all_fc = False):
        super(capMDNet, self).__init__()
        self.K = K
        self.roi_size = roi_size
        self.num_predictions = num_predictions
        self.all_fc = all_fc
        self.spatial_scale = spatial_scale
        if not self.all_fc:
            self.layers = nn.Sequential(OrderedDict([
                    ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=3),
                                            nn.ReLU(),
                                            LRN(),
                                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                    ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                            nn.ReLU(),
                                            LRN(),
                                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                    ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Dropout(0.5)))])) # Add a Dropout layer
            
            # Also we can add another conv layer to reduce the map dimension
            #nn.Conv2d
            # ROI extraction based on the feature maps and bbox coordinate in the image  
            #self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
            #self.roi = _RoIPooling(self.roi_size, self.roi_size, self.spatial_scale)#another version of roipooling
            self.roi = RoIAlignAvg(self.roi_size, self.roi_size, self.spatial_scale)
        
        
            # Capsule parts
            self.primary = nn.Sequential(OrderedDict([
                    ('capPrimary', CapsuleLayer(in_unit=0,
                                in_channel=num_input_primary_capsule_channel,
                                num_unit=num_primary_unit,
                                unit_size=primary_unit_size, 
                                use_routing=False,
                                num_routing=num_routing,
                                out_channel = num_output_primary_capsule_channel,
                                status = network_status,
                                regrouping_type = regrouping_type,
                                group_attention = group_attention))]))
            
            self.branches = nn.ModuleList([nn.Sequential(CapsuleLayer(in_unit=num_primary_unit,
                                       in_channel=primary_unit_size, 
                                       num_unit=num_predictions,     
                                       unit_size=output_unit_size,   
                                       use_routing=True,
                                       num_routing=num_routing, 
                                       status = network_status,
                                       high_cap_conv = high_cap_conv,
                                       single_conv = single_conv,
                                       noTM = noTM,
                                       penalty_attention = penalty_attention,
                                       fc = fc)) for _ in range(K)])
        else:
            self.layers = nn.Sequential(OrderedDict([
                    ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=3),
                                            nn.ReLU(),
                                            LRN(),
                                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                    ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                            nn.ReLU(),
                                            LRN(),
                                            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))),
                    ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(512), # the difference
                                            nn.ReLU(),
                                            nn.Dropout(0.5)))])) # Add a Dropout layer
            self.roi = RoIAlignAvg(self.roi_size, self.roi_size, self.spatial_scale)
            self.cap_fc = nn.Sequential(OrderedDict([
                    ('cap_fc', nn.Sequential(nn.Linear(512*7*7,2
                                                       )))]))
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        if not self.all_fc:
            for name, module in self.primary.named_children():
                append_params(self.params, module, name)
            for k, module in enumerate(self.branches):
                append_params(self.params, module, 'cap_%d'%(k))
        else:
            for name, module in self.cap_fc.named_children():
                append_params(self.params, module, name)

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    def stop_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, rois=None, im_indices=None, k=0, in_layer='conv1', out_layer='capsule',fea_view = False):
        if rois is not None:
            im_indices = totensor(im_indices).float()
            rois = totensor(rois).float()
            indices_and_rois = torch.cat([im_indices[:, None], rois], dim=1)
        conv3_fea = []
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    conv3_fea = x
                    x = self.roi(x, indices_and_rois) # [len(im_indices),512,7,7]
                if name == out_layer:
                    if fea_view & (name == 'conv3'):
                        x = x.view(len(im_indices),-1)
                    return x
        if not self.all_fc:
            x = self.primary(x)
            cap_out = self.branches[k](x)
        else:
            #x = squash(x, dim=1) # [-,512,7,7]
            x = x.view(x.shape[0],-1)
            cap_out = self.cap_fc(x).unsqueeze(2).unsqueeze(3)# [batch_size, 2, 1, 1]
        if out_layer=='capsule':
            x = torch.sqrt((cap_out**2).sum(dim=2)).view(-1,self.num_predictions) # [len(im_indices),2]
            return x, conv3_fea
    
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
        if not self.all_fc:
            primary_layer = states['primary_layer']
            branch_layer = states['prediction_layer']
            self.primary.load_state_dict(primary_layer)
            self.branches.load_state_dict(branch_layer)
        else:
            fc_layer = states['fc_layer']
            self.cap_fc.load_state_dict(fc_layer)
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:,1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        
        return loss
    
class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()
    def forward(self, score, target):
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - score, zero)**2 # [batch_size, 2]
        max_right = torch.max(score - m_minus, zero)**2# [batch_size, 2]
        t_c = target.cuda()
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        loss = l_c.sum(dim=1)
        return loss.sum()
    
def one_hot_encode(pos_n, neg_n, length):
    batch_s = pos_n + neg_n 
    one_hot_vec = torch.zeros(batch_s, length)

    for i in range(batch_s):
        if i < pos_n:
            one_hot_vec[i, 1] = 1.0
        else:
            one_hot_vec[i, 0] = 1.0

    return one_hot_vec


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.item()