 #!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

nstates_plus_1 = [64, 64, 256]
nstates_plus_2 = [128, 128, 256]
nstates_plus_3 = [256, 512, 1024, 1024, 1024]

vlad_dim_out = 128 * 8
 

class PointNet_Plus(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=10, dim=512, normalize_input=True):
        super(PointNet_Plus, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost

        self.normalize_input = normalize_input
        self.pooling = opt.pooling

        if self.pooling == 'concatenation':
            self.dim_out = 1024

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, 64), stride=1)
        )


        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        

        self.my_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level2, 1), stride=1))
        self.gobaol_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level2*self.gost, 1), stride=1))
        self.netR_FC = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties

    def forward(self, xt, yt, loss_mode=0):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------
        B, d, N, k = xt.shape
        xt = self.net3DV_1(xt)
        xt = torch.cat((yt, xt), 1)#.squeeze(-1)
        xt_local = self.net3DV_3(xt)  # (gost * batch) * 1024 * 64 *1
        
        xt = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)
        x = xt.clone()  # batch *1024
        
        # x_global = xt_local.reshape(self.gost, -1, 1024, 64).permute(1, 2, 0, 3).reshape(-1, 1024, self.gost*64, 1)
        # x_global = self.gobaol_max_pool(x_global).squeeze(-1).squeeze(-1)

        x = self.netR_FC(x)  # batch * 128
        # x_global = self.netR_FC(x_global)

        # x_nor = F.normalize(x, p=2, dim=1)  # intra normalize x:(5 * batch) * 128
        # code = self.mapping(x_nor)  # code : (5*batch ) *centers

        return x#, 0, 0, 0
    
    # def forward(self, xt, yt, loss_mode=0):
    
    #     # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
    #     ###----motion stream--------
    #     B, d, N, k = xt.shape
    #     xt = self.net3DV_1(xt)
    #     xt = torch.cat((yt, xt), 1)#.squeeze(-1)
    #     xt_local = self.net3DV_3(xt)  # (gost * batch) * 1024 * 64 *1
        
    #     xt = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)
    #     x = xt.clone()  # batch *1024
        
    #     x_global = xt_local.reshape(self.gost, -1, 1024, 64).permute(1, 2, 0, 3).reshape(-1, 1024, self.gost*64, 1)
    #     x_global = self.gobaol_max_pool(x_global).squeeze(-1).squeeze(-1)

    #     x = self.netR_FC(x)  # batch * 128
    #     x_global = self.netR_FC(x_global)

    #     x_nor = F.normalize(x, p=2, dim=1)  # intra normalize x:(5 * batch) * 128
    #     code = self.mapping(x_nor)  # code : (5*batch ) *centers

    #     return x, code, x_nor, x_global
    
    
    
class PointNet_Plus_fine(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=10, dim=512, sample_num_level1=32, knn_K = 128, normalize_input=True):
        super(PointNet_Plus_fine, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost

        self.normalize_input = normalize_input
        self.pooling = opt.pooling

        if self.pooling == 'concatenation':
            self.dim_out = 1024

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, self.knn_K), stride=1)
        )


        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3 + nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            # nn.MaxPool2d((self.sample_num_level2, 1), stride=1)
            # B*1024*1*1
        )
        

        self.my_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level1, 1), stride=1))
        self.gobaol_max_pool = nn.Sequential(nn.MaxPool2d((self.sample_num_level1*self.gost, 1), stride=1))
        self.netR_FC = nn.Sequential(
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties

    
    def forward(self, xt, yt, loss_mode=0):
    
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------
        B, d, N, k = xt.shape
        xt = self.net3DV_1(xt)
        xt = torch.cat((yt, xt), 1)#.squeeze(-1)
        xt_local = self.net3DV_3(xt)  # (gost * batch) * 1024 * 64 *1
        
        xt = self.my_max_pool(xt_local).squeeze(-1).squeeze(-1)
        x = xt.clone()  # batch *1024
        
        x_global = xt_local.reshape(self.gost, -1, 1024, self.sample_num_level1).permute(1, 2, 0, 3).reshape(-1, 1024, self.gost*self.sample_num_level1, 1)
        x_global = self.gobaol_max_pool(x_global).squeeze(-1).squeeze(-1)

        x = self.netR_FC(x)  # batch * 128
        x_global = self.netR_FC(x_global)

        x_nor = F.normalize(x, p=2, dim=1)  # intra normalize x:(5 * batch) * 128
        code = self.mapping(x_nor)  # code : (5*batch ) *centers

        return x, code, x_nor, x_global
    

nstates_1 = [64, 128, 256]
nstates_3 = [256, 512, 1024, 1024, 1024]
        
slow_1 = [16, 64, 128]
slow_3 = [128, 256, 512, 1024, 1024]

class PointNet_Slow_Fast(nn.Module):
    def __init__(self, opt, num_clusters=64, gost=10, dim=512, normalize_input=True):
        super(PointNet_Slow_Fast, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM  # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        self.batch = opt.batchSize
        self.dim = dim
        self.num_clusters = num_clusters
        self.gost = gost

        self.normalize_input = normalize_input
        self.pooling = opt.pooling

        # if self.pooling == 'concatenation':
        self.dim_out_slow = slow_3[2] 
        self.dim_out_fast =nstates_3[2]

        self.net3DV_slow1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, slow_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(slow_1[0], slow_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(slow_1[1], slow_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, 128), stride=1)
        )


        self.net3DV_slow3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(slow_1[2], slow_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(slow_3[0], slow_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(slow_3[1], slow_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(slow_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((64, 1), stride=1)
            # B*1024*1*1
        )
        
        
        self.net3DV_fast1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_1[0], nstates_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_1[1], nstates_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, 64), stride=1)
        )
        
        
        self.net3DV_fast3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(nstates_plus_2[2], nstates_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_3[0], nstates_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_3[1], nstates_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((64, 1), stride=1)
            # B*1024*1*1
        )

        self.netR_FC_slow = nn.Sequential(
            nn.Linear(self.dim_out_slow, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
        
        self.netR_FC_fast = nn.Sequential(
            nn.Linear(self.dim_out_fast, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.dim),
        )
        
        self.fast_to_slow = nn.Sequential(
            nn.Conv2d(nstates_1[2], slow_1[2], kernel_size=(1, 1)))
        
        self.slow_to_fast = nn.Sequential(
            nn.Conv2d(slow_1[2], nstates_1[2], kernel_size=(1, 1)))
   

        self.mapping = nn.Linear(self.dim, self.num_clusters, bias=False)  # mapping weight is the properties

    def forward(self, x_slow, x_fast):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1
        ###----motion stream--------
        B, d, N, k = x_slow.shape
        x_slow = self.net3DV_slow1(x_slow)
        x_fast = self.net3DV_fast1(x_fast)
        
        x_sf = self.slow_to_fast(x_slow)
        x_fs = self.fast_to_slow(x_fast)
        
        x_slow = torch.cat((x_slow, x_fs), dim =2)
        x_fast = torch.cat((x_fast, x_sf), dim =2)
        
        x_slow = self.net3DV_slow3(x_slow).squeeze().squeeze()  # (gost * batch) * 1024 * 64 *1      
        x_fast = self.net3DV_fast3(x_fast).squeeze().squeeze()  # (gost * batch) * 1024 * 64 *1
        # x = torch.cat((x_fast, x_slow), dim =1)
        
        x_fast = self.netR_FC_fast(x_fast)  # batch * 128
        x_slow = self.netR_FC_slow(x_slow)
    
        # x_nor = F.normalize(x, p=2, dim=1)  # intra normalize x:(5 * batch) * 128
        # code = self.mapping(x_nor)  # code : (5*batch ) *centers

        return x_fast, x_slow#, x_nor



def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            # dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, dim=512, K=9600, m=0.9, T=1, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.opt = opt

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = PointNet_Plus(self.opt)
        self.encoder_k = PointNet_Plus(self.opt)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x, y):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # # compute query features
        # q = self.encoder_q(x[0:self.opt.batchSize], y[0:self.opt.batchSize])  # queries: NxC

        # # q = nn.functional.normalize(q, dim=1)

        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     self._momentum_update_key_encoder()  # update the key encoder

        #     k = self.encoder_k(x[self.opt.batchSize:], y[self.opt.batchSize:])  # keys: NxC
        #     # k = nn.functional.normalize(k, dim=1)


        # # compute logits
        # # Einstein sum is more intuitive
        # # positive logits: Nx1

        # # num_crop  = 10  
        # # l_pos = torch.Tensor(num_crop, self.opt.batchSize, 1).cuda()

        # # for lo_i in range(0,num_crop):
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
  
        # l_neg2 = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits_p= torch.cat([l_pos, l_neg2], dim=1)
        # # labels: positive key indicators
        # labels = torch.zeros(logits_p.shape[0], dtype=torch.long).cuda()
        
        # # apply temperature
        # logits_p /= self.T

        # # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        # return logits_p, labels, q

        q = self.encoder_q(x, y)  # queries: NxC
        return q


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output





