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
# import gpu_utils as g
import numpy as np

# *******************************mode change to use dgcnn
import cn3d_model_conbag as MM
from cn3D_data_set import NTU_RGBD_new
# from cn3d_data_load import  deal_data_4 as deal_data_4

# from utils import group_points,group_points_pro

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_my import group_points_3DV

id = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str,
                        default='../ntu/ntu60_new2/raw/',
                        help='preprocess folder')
    parser.add_argument('--depth_path', type=str, default='/data/data3/wangyancheng/ntu120dataset/',
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
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

    parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default=512, help='number of sample points')

    # class change to 2 class
    parser.add_argument('--Num_Class', type=int, default=128, help='number of outputs')

    parser.add_argument('--knn_K', type=int, default=64, help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default=64, help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default=64, help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.16,
                        help='square of radius for ball query in level 1')  # 0.025 -> 0.05 for detph
    parser.add_argument('--ball_radius2', type=float, default=0.25,
                        help='square of radius for ball query in level 2')  # 0.08 -> 0.01 for depth

    parser.add_argument('--save_feature_dir', type=str,
                        default='../ntu/ntu60_new2/features/',
                        help='fe_output folder')

    parser.add_argument('--save_label_dir', type=str,
                        default='../ntu/ntu60_new2/labels/',
                        help='fe_output folder')

    parser.add_argument('--branch_choose', type=str,
                        default='0', help='choose which branch motion or apperance')

    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # data use 3dv point cloud generate from NTU120 to train contrast net
    # data struct is defined as :
    data_train = NTU_RGBD_new(root_path='../ntu/3DV_ntu60/raw', opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=True,
                          validation=False,
                          test=False,
                          Transform=False,
                          DATA_CROSS_SET=False
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=8)

    #

    data_val = NTU_RGBD_new(root_path='../ntu/3DV_ntu60/raw', opt=opt,
                        # opt.root_path
                        DATA_CROSS_VIEW=True,
                        full_train=False,
                        validation=False,
                        test=True,
                        Transform=False,
                        DATA_CROSS_SET=False
                        )
    val_loader = DataLoader(dataset=data_val, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=8)

    feture_mode = 2
    num_crop = 10
    kerrner = 3
    drop_dim = 0

    netR = MM.PointNet_Plus(opt)
    print('loding the  mode...................')
    model_name = '../ntu/ntu60_new2/model/corr_GL_.pth'  # have_motion_model_     no_motion_model_   no_z_model_
    netR.load_state_dict(torch.load(model_name))
    print(model_name, 'loding finish...................')
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()

    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # switch to train mode
    torch.cuda.synchronize()
    netR.eval()
    save_path = '/data/data1/ntu/feature/ntu60/cv/G_L/motion_corr/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('now start extrat train data fetaure... ')
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader, 0)):


            # points, v_name, key_points_path, key_points, label,time_seg1, time_seg2, time_seg3,time_seg4 = data
            # data1 = deal_data_4(points, key_points,time_seg1, time_seg2, time_seg3,time_seg4, num_crop)
            # data1 = torch.from_numpy(data1)
            
            
            out_points, v_name, label = data
            B, G, N, D = out_points.shape
            data1 = out_points.permute(1, 0, 2, 3).reshape(-1,N, D)
           
            data1 = data1.type(torch.FloatTensor).cuda()
            opt.ball_radius = opt.ball_radius + random.uniform(-0.02, 0.02)
            xt, yt = group_points_3DV(data1, opt)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1
            torch.cuda.synchronize()
            
            x,  _, _,x_global= netR(xt, yt)
        
            x = torch.cat((x, x_global), dim =0)
            feature1 = x.cpu().detach().numpy()
            save_single_feature(feature1, save_path, v_name)
        


        print('now start extrat test data fetaure....')

        # feature_f = open(opt.save_feature_dir + str(vote_idx) + '.txt', 'w')
        # need to rewite data_get
        for i, data in enumerate(tqdm(val_loader, 0)):

            # points, v_name, key_points_path, key_points, label,time_seg1, time_seg2, time_seg3,time_seg4 = data
            # data1 = deal_data_4(points, key_points,time_seg1, time_seg2, time_seg3,time_seg4, num_crop)
            # data1 = torch.from_numpy(data1)
            
            
            out_points, v_name, label = data
            B, G, N, D = out_points.shape
            data1 = out_points.permute(1, 0, 2, 3).reshape(-1,N, D)
           
            data1 = data1.type(torch.FloatTensor).cuda()
            opt.ball_radius = opt.ball_radius + random.uniform(-0.02, 0.02)
            xt, yt = group_points_3DV(data1, opt)  # batch_size, 8, 512, 64    batch_size, 3, 512, 1
            torch.cuda.synchronize()

            torch.cuda.synchronize()
           
            x, _, _, x_global= netR(xt, yt)
        
            x = torch.cat((x, x_global), dim =0)
            feature1 = x.cpu().detach().numpy()
            save_single_feature(feature1, save_path, v_name)


def save_single_feature(feature, save_path, name, num_crop = 11):
    feature = feature.reshape(num_crop, -1 , 512).transpose(1, 0, 2).reshape(-1, num_crop * 512)
    for batch_i in range(feature.shape[0]):
        f_p = save_path + name[batch_i] + '.npy'
        np.save(f_p, feature[batch_i])





if __name__ == '__main__':
    main()

