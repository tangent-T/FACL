# -*- coding: utf-8 -*-
import logging

import torch
import os
import tqdm
import shutil
import collections
import argparse
import random
import time
import numpy as np
import torch.nn.functional as F

# *******************************mode change to use dgcnn
import cn3d_model_conbag as MODELL
from cn3D_data_set import NTU_RGBD, NTU_RGBD_new
# from cn3d_data_load import deal_data_new

# from utils import group_points,group_points_pro

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_my import group_points_3DV

id = 2
kener = [2, 4, 5]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def grouping(features_groupDis1, features_groupDis2, T, k_eigen, clusters, num_iters):
    # print(features_groupDis1.size())
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # K-way normalized cuts or k-Means. Default: k-Means

    cluster_label1, centroids1 = KMeans(features_groupDis1, clusters, num_iters)
    cluster_label2, centroids2 = KMeans(features_groupDis2, clusters, num_iters)

    # group discriminative learning
    affnity1 = torch.mm(features_groupDis1, centroids2.t())
    CLD_loss = criterion(affnity1.div_(T), cluster_label2)

    affnity2 = torch.mm(features_groupDis2, centroids1.t())
    CLD_loss = (CLD_loss + criterion(affnity2.div_(T), cluster_label1))/2
    return CLD_loss

def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c



def main(args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=4, help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default=3, help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation',
                        help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60',
                        help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str,
                        default='../ntu/ntu60_new2/raw/',
                        help='preprocess folder')
    parser.add_argument('--depth_path', type=str, default='',
                        help='raw_depth_png')

    parser.add_argument('--save_root_dir', type=str, default='../ntu/ntu60_new2/model/',
                        help='output folder')
    parser.add_argument('--model', type=str, default='', help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    # defined by the DGCNN
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')

    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default=512, help='number of sample points')

    # class change to 2 class
    parser.add_argument('--Num_Class', type=int, default=512, help='number of outputs')

    parser.add_argument('--knn_K', type=int, default=64, help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default=64, help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default=64, help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.16,
                        help='square of radius for ball query in level 1')  # 0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.25,
                        help='square of radius for ball query in level 2')  # 0.08 -> 0.01 for depth
    parser.add_argument('--ex_feature', type=int, default=7,
                        help='motion data or appearance data')  # 0.08 -> 0.01 for depth

    parser.add_argument('--save_feature_dir', type=str,
                        default='../ntu/ntu60_new2/features/',
                        help='fe_output folder')
    parser.add_argument('--save_label_dir', type=str,
                        default='../ntu/ntu60_new2/labels/',
                        help='fe_output folder')
    parser.add_argument('--branch_choose', type=str,
                        default='1')

    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        os.makedirs(opt.save_root_dir)
    except OSError:
        pass
    # use gpu set 0&1
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                        filename='../ntu/ntu60_new2/30_0425.log', level=logging.INFO)
    logging.info('======================================================')

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # data use 3dv point cloud generate from NTU120 to train contrast net
    # data struct is defined as :
    data_train = NTU_RGBD_new(root_path='../ntu/3DV_ntu60/reslution/Resolution10/raw/', opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=True,
                          validation=False,
                          test=False,
                          Transform=False
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=16)

    netR = MODELL.PointNet_Plus(opt)
    print('loding the  mode...................')
    
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)

    MODE = 1
    SAVE_MODE = 2

    queue = None
    use_the_queue = False
    epoch_queue_starts = 10
    num_crop = 10
    queue_length = opt.batchSize * 32

    fix_weight = np.ones((opt.batchSize, opt.batchSize)) - np.identity(opt.batchSize)
    fix_weight_loss = np.tile(fix_weight, (1, num_crop))

    fix_fusion_loss = np.tile(fix_weight, (1, 27))
    fix_weight_circle = np.tile(fix_weight, (1, (num_crop) * (num_crop - 1)))


    fix_weight_loss = torch.from_numpy(fix_weight_loss)
    fix_weight_loss = fix_weight_loss.type(torch.FloatTensor).cuda()

    fix_weight_circle = torch.from_numpy(fix_weight_circle)
    fix_weight_circle = fix_weight_circle.type(torch.FloatTensor).cuda()

    print('\n MODE is :', MODE, 'save mode is : ', SAVE_MODE)

    for epoch in range(0, opt.nepoch):
        # switch to train mode
        torch.cuda.synchronize()
        netR.train()
        loss_sigma = 0.0

        loss_mode = 1
        softmax = torch.nn.Softmax(dim=1).cuda()
        if queue_length > 0 and epoch >= epoch_queue_starts and queue is None:
            queue = torch.zeros(
                num_crop-1,
                queue_length,
                512
            ).cuda()

        # need to rewite data_get
        for i, data in enumerate(tqdm(train_loader, 0)):
            out_points, v_name, label = data
            B, G, N, D = out_points.shape
            out_points = out_points.permute(1, 0, 2, 3).reshape(-1,N, D)
            
            data1 = out_points.type(torch.FloatTensor).cuda()
            opt.ball_radius = opt.ball_radius + random.uniform(-0.02, 0.02)
            xt, yt = group_points_3DV(data1, opt)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1

            torch.cuda.synchronize()

            x, code, x_nor, x_global= netR(xt, yt, loss_mode)
            # print(x_nor.shape, queue.shape)
            ## cluster part
            loss_swa = 0
            swa_if = 0
            if swa_if == 1:
                for crop_id in range(num_crop-1):  # 4 kinds positive samples  crop_id is the all kinds of augmentation data
                    with torch.no_grad():
                        po = code[opt.batchSize * crop_id:opt.batchSize * (crop_id + 1), :]  # batch * centers
                        # calculate the queue to the code according to the properties
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[crop_id, -1, :] == 0):
                                use_the_queue = True
                                # aa= torch.mm(queue[i], netR.module.mapping.weight.t())
                                po = torch.cat((torch.mm(queue[crop_id], netR.module.mapping.weight.t()), po))
                            # fill the queue
                            queue[crop_id, opt.batchSize:, :] = queue[crop_id, :-opt.batchSize, :].clone()
                            queue[crop_id, 0:opt.batchSize, :] = x_nor[crop_id * opt.batchSize: (crop_id + 1) * opt.batchSize, :]
    
                        po = po / 0.03
                        aa = po.detach().cpu().numpy()
                        po = torch.exp(po).t()  # centers * batch
                        q = MODELL.distributed_sinkhorn(po, 3)[-opt.batchSize:]
                    subloss = 0
                    for v in np.delete(np.arange(num_crop-1), crop_id):
                        p = softmax(code[opt.batchSize * v: opt.batchSize * (v + 1)] / 0.1)
                        subloss = subloss - torch.mean(torch.sum(q * torch.log(p), dim=1))
                        ss = subloss.detach().cpu().numpy()
                    loss_swa = loss_swa + subloss
                loss_swa = loss_swa / (num_crop-1)
            
            global_if = 1
            if global_if ==1:
                
                l_pos = torch.Tensor(num_crop, opt.batchSize, 1).cuda()
                loss_all = torch.Tensor(num_crop, 1).cuda()
    
                for lo_i in range(0,num_crop):
                    l_pos[lo_i] = torch.einsum('nc,nc->n', [x_global, x[(lo_i) * opt.batchSize:(lo_i+1) * opt.batchSize, :]]).unsqueeze(-1)
    
                k1 = x.clone().permute(1, 0)
                l_neg2 = torch.einsum('nc,ck->nk', [x_global, k1])  # batch * (5*batch)
                l_neg2 = torch.mul(l_neg2, fix_weight_loss)

                l_neg2 = l_neg2.repeat(num_crop,1,1)
    
                logits_p= torch.cat([l_pos, l_neg2], dim=2)
    
                # labels: positive key indicators
                labels = torch.zeros(logits_p.shape[1], dtype=torch.long).cuda()
                for loss_i in range(num_crop):
                    loss_all[loss_i] = criterion(logits_p[loss_i], labels)
    
                loss_c = (loss_all.sum())


            circle_if = 1

            #loss circle part
            if circle_if == 1:
                l_pos_circle_all = torch.Tensor(num_crop-1, opt.batchSize, 1).cuda()
                l_neg_circle_all = torch.Tensor(num_crop-1, opt.batchSize, num_crop*opt.batchSize).cuda()

                order_idex = np.arange(0, num_crop, 1)  # 0~9
                np.random.shuffle(order_idex)
                loss_circle_all = torch.Tensor(num_crop-1).cuda()
                for l_pos1_circle_i in range(num_crop-1):
                    l_pos_circle_all[l_pos1_circle_i] = torch.einsum('nc,nc->n',[x[order_idex[l_pos1_circle_i] * opt.batchSize:(order_idex[l_pos1_circle_i] + 1) * opt.batchSize, :],x[order_idex[l_pos1_circle_i+1] * opt.batchSize:(order_idex[l_pos1_circle_i+1] + 1) * opt.batchSize,:]]).unsqueeze(-1)
    
                for l_neg_circle_i in range(num_crop-1):
                    l_neg_circle_all[l_neg_circle_i] = torch.einsum('nc,ck->nk', [x[order_idex[l_neg_circle_i] * opt.batchSize:(order_idex[l_neg_circle_i] + 1) * opt.batchSize, :],x.clone().permute(1, 0)])

                l_neg_circle = l_neg_circle_all.permute(1,0,2).reshape(opt.batchSize, -1)
                l_neg_circle = torch.mul(l_neg_circle, fix_weight_circle)

                l_neg_circle = l_neg_circle.repeat(num_crop-1, 1 ,1)
                logits_p_circle = torch.cat([l_pos_circle_all, l_neg_circle], dim=2)
    
                # labels: positive key indicators
                labels = torch.zeros(logits_p_circle.shape[1], dtype=torch.long).cuda()
                for loss_circle_i in range(num_crop-1):
                    loss_circle_all[loss_circle_i] = criterion(logits_p_circle[loss_circle_i], labels)
                loss_circle = loss_circle_all.sum()
            

            cld_if = 0
            loss_CLD = 0
            CLD_start = 0
            if cld_if ==1 and epoch>=CLD_start:
                loss_CLD_all = torch.Tensor(num_crop-4).cuda()
                for CLD_i in range(num_crop-4):
                    loss_CLD_all[CLD_i] = grouping(x_nor[CLD_i * opt.batchSize: (CLD_i+3) * opt.batchSize], x_nor[(CLD_i+1) * opt.batchSize: (CLD_i+4) * opt.batchSize], 0.05, 10, 60, 5) #k_eigen, clusters, num_iters
                loss_CLD = loss_CLD_all.sum()

            
            loss =  loss_circle   +0.6 * loss_swa  +loss_CLD + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            torch.cuda.synchronize()
            loss_sigma += loss.item()
            # print(loss_circle.item(), loss_global.item())

        logging.info('{} --epoch{} ==Average loss:{}'.format('Valid', epoch, loss_sigma / (i + 1)))
        print('epoch:', epoch, 'loss mode is :', loss_mode, '--loss:', loss_sigma / (i + 1))
        if epoch % 5 == 0:
            torch.save(netR.module.state_dict(), '%s/corr_GL_appereance_%d.pth' % (opt.save_root_dir, epoch)) # no_ss_implify_


if __name__ == '__main__':
    main()


