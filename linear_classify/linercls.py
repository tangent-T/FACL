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
from fc_model import Final_FC
from dataset_of_lin import LIner_NTU

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


id = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main(args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--dataset', type=str, default='ntu120',help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--root_path', type=str,default='../ntu/3DV_ntu60/reslution/Resolution60/raw/',help='preprocess folder')

    parser.add_argument('--save_root_dir', type=str, default='../ntu/3DV_ntu60/model/',help='output folder')
    parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')

    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    parser.add_argument('--motion_feature', type=str, default='', help='learning rate decay')
    parser.add_argument('--appreance_feature', type=str, default='', help='learning rate decay')
 

    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.main_gpu)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    data_train = LIner_NTU(root_path=opt.root_path, opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=True,
                          validation=False,
                          test=False,
                          DATA_CROSS_SET=False
                          )

    # get all the data fold into batches
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=8)


    data_val = LIner_NTU(root_path=opt.root_path, opt=opt,
                          # opt.root_path
                          DATA_CROSS_VIEW=True,
                          full_train=False,
                          validation=False,
                          test=True,
                          DATA_CROSS_SET=False
                          )
    val_loader = DataLoader(dataset=data_val, batch_size=opt.batchSize, drop_last=True, shuffle=False, num_workers=8)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    netR = Final_FC()
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()

    optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), eps=1e-06)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename='/home/tanbo/cn3d_pointnet_kmeans/liner_cls_all/0815fc.log', level=logging.INFO)
    # logging.info('======================================================')
    best = 0

    torch.cuda.synchronize()
    print('now start extrat train data fetaure... ')
    for epoch in range(opt.nepoch):
        netR.train()
        loss_sigma = 0
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for i, data in enumerate(tqdm(train_loader, 0)):
            features, labels = data
            features = features.type(torch.FloatTensor)#, labels.type(torch.FloatTensor)
            features, labels = features.squeeze().cuda(), labels.squeeze().cuda()
            labels = labels.reshape(-1)
            output = netR(features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(epoch)
            torch.cuda.synchronize()
            loss_sigma = loss_sigma+loss.item()

            acc1, acc5 = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], features.size(0))
            top5.update(acc5[0], features.size(0))

        logging.info('{} --epoch{} ==Average loss:{}    top1: {}'.format('train', epoch, loss_sigma / (epoch + 1), top1.avg))
        print('epoch:', epoch, 'loss:', loss_sigma/(epoch+1), 'train top1:', top1.avg)
        
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        # a = netR.module.fc.weight.t()
        # a = a.detach().cpu().numpy()
        # a = np.sum(np.abs(a), axis=1)
        # idex = np.argsort(-a)
        # print(idex[:100])
        netR.eval()
        if epoch>15:
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader, 0)):
                    features, labels = data
                    features = features.type(torch.FloatTensor)#, labels.type(torch.FloatTensor)
                    features, labels = features.squeeze().cuda(), labels.squeeze().cuda()
                    output = netR(features)

                    acc1, acc5 = accuracy(output, labels, topk=(1, 1))
                    top1.update(acc1[0], features.size(0))
                    top5.update(acc5[0], features.size(0))

                logging.info('{} --epoch{}   top1: {}'.format('Valid', epoch, top1.avg))
                print('epoch:', epoch,  'test top1:', top1.avg)

                # if top1.avg> best:
                #     best = top1.avg
                #     torch.save(netR.module.state_dict(), '%s/FC_model_0930_%d.pth' % ('../ntu/FC_model/', epoch))

    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()