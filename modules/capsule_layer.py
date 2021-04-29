"""
The capsule_layer.py is referenced from https://github.com/cedrickchee/capsule-net-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils


class CapsuleLayer(nn.Module):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing, num_routing, out_channel = None, status ='Train', 
                 regrouping_type = None, group_attention = None, high_cap_conv = None, single_conv = None, 
                 noTM = None, penalty_attention = None, fc = None):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        
        self.status = status

        if not self.use_routing:
            self.out_channel = out_channel
            self.regrouping_type = regrouping_type
            self.group_attention = group_attention
            self.group_num = self.out_channel/self.num_unit
            self.conv_units =nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2))
            if self.group_attention:
                self.attention = nn.Sequential(nn.AvgPool2d(kernel_size = 3),
                        nn.Conv2d(self.out_channel, self.group_num, kernel_size=1, stride=1,groups=self.group_num),
                        nn.Sigmoid()) 
        else:
            self.fc = fc
            self.high_cap_conv = high_cap_conv
            self.penalty_attention = penalty_attention
            self.unit_size = unit_size
            self.num_routing = num_routing
            self.single_conv = single_conv
            self.noTM = noTM
            if (not self.high_cap_conv) & (not self.noTM):
                self.weight = nn.Parameter(torch.randn(1, self.in_channel, self.num_unit, self.unit_size, self.in_unit))
            elif self.high_cap_conv & (not self.noTM): 
                if self.single_conv:
                    self.weight_conv = nn.Sequential(nn.Conv2d(self.in_channel*self.num_unit, self.in_channel*self.num_unit,kernel_size=(1,1), stride=1,
                                             groups=self.in_channel*self.num_unit))
                else:
                    self.weight_conv_1 = nn.Sequential(nn.Conv2d(self.in_channel*self.num_unit, self.in_channel*self.num_unit,kernel_size=(4,1), stride=1,
                                             groups=self.in_channel*self.num_unit))
                    self.weight_conv_2 = nn.Sequential(nn.Conv2d(self.in_channel*self.num_unit, self.in_channel*self.num_unit,kernel_size=(4,1), stride=1,
                                             groups=self.in_channel*self.num_unit))
                    self.weight_conv_3 = nn.Sequential(nn.Conv2d(self.in_channel*self.num_unit, self.in_channel*self.num_unit,kernel_size=(4,1), stride=1,
                                             groups=self.in_channel*self.num_unit))
                    self.weight_conv_4 = nn.Sequential(nn.Conv2d(self.in_channel*self.num_unit, self.in_channel*self.num_unit,kernel_size=(4,1), stride=1,
                                             groups=self.in_channel*self.num_unit))
            if not self.fc:
                if self.penalty_attention:
                    self.penalty = nn.Parameter(torch.ones(1, 1, num_unit, unit_size,1))
            else:
                self.fully_connected = nn.Sequential(nn.Linear(self.in_channel*self.num_unit*self.unit_size, self.num_unit))

    def forward(self, x):
        if not self.use_routing:
            return self.no_routing(x)
        else:
            return self.routing(x)
    
    def regrouping_local(self, unit):
        return unit 
    
    def regrouping_adjacent(self, unit):
        unit = torch.cat((unit[:,(-1)*self.num_unit/2:,:,:], unit[:,:(-1)*self.num_unit/2,:,:]),1)
        return unit
    
    def regrouping_shuffle(self, unit, batch_size, spatial_size):
        unit = unit.view(batch_size, self.num_unit, self.group_num, spatial_size, spatial_size)
        unit = unit.transpose(1,2).contiguous()
        unit = unit.view(batch_size, self.out_channel, spatial_size, spatial_size)
        return unit
    
    def no_routing(self, x):
        batch_size = x.size(0)
        unit = self.conv_units(x)
        spatial_size = unit.size(2)
        if self.regrouping_type == 'local':
            unit = self.regrouping_local(unit)
        elif self.regrouping_type == 'adjacent': 
            unit = self.regrouping_adjacent(unit)
        elif self.regrouping_type == 'shuffle':
            unit = self.regrouping_shuffle(unit, batch_size, spatial_size)
        
        if self.group_attention:
            attention_weight = self.attention(unit)
            attention_weight = torch.stack([attention_weight] * self.num_unit, dim=1)
        unit = unit.view(batch_size, self.group_num, self.num_unit, spatial_size, spatial_size).transpose(1,2).contiguous()
        unit = attention_weight*unit
        unit = unit.view(batch_size, self.num_unit, -1)
        return utils.squash(unit, dim=2)
    
    def routing(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        if (not self.high_cap_conv) & (not self.noTM):
            x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
            batch_weight = torch.cat([self.weight] * batch_size, dim=0)
            u_hat = torch.matmul(batch_weight, x)
        elif self.high_cap_conv & (not self.noTM):
            if self.single_conv: 
                x = torch.stack([x] * self.num_unit, dim=1).unsqueeze(4)
                x = x.view(batch_size,-1,self.in_unit,1) 
                u_hat = self.weight_conv(x) 
                u_hat = (u_hat.view(batch_size, self.num_unit, self.in_channel,u_hat.size(2),1)).transpose(1,2).contiguous()
            else:
                x = torch.stack([x] * self.num_unit, dim=1).unsqueeze(4)
                x = x.view(batch_size,-1,self.in_unit,1)
                u_hat_1 = self.weight_conv_1(x)
                u_hat_2 = self.weight_conv_2(x)
                u_hat_3 = self.weight_conv_3(x)
                u_hat_4 = self.weight_conv_4(x) 
                u_hat = torch.cat((u_hat_1,u_hat_2,u_hat_3,u_hat_4), 2) 
                u_hat = (u_hat.view(batch_size, self.num_unit, self.in_channel,u_hat.size(2),1)).transpose(1,2).contiguous()
        elif self.noTM:
            u_hat = x

        if not self.fc:
            b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
            b_ij = b_ij.cuda()
            if self.penalty_attention:
                penalty = torch.cat([self.penalty] * batch_size, dim=0)
            for iteration in range(self.num_routing):
                c_ij = F.softmax(b_ij, dim=2)
                c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                v_j = utils.squash(s_j, dim=3)
                
                if self.penalty_attention:
                    v_j = v_j * penalty
                v_j1 = torch.cat([v_j] * self.in_channel, dim=1)
                u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
                b_ij = b_ij + u_vj1
            return v_j.squeeze(1)
        else:
            v = u_hat.view(batch_size, -1)
            v = self.fully_connected(v)
            return v.unsqueeze(2).unsqueeze(3)

