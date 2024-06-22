import os
import tqdm
import torch
import re
import collections
import imageio
import random

from tqdm import tqdm

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import scipy.io as sio

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
# rose@ntu.edu.sg
sample_num_level1 = 512
sample_num_level2 = 128
NUM_POINT = 512
TRAIN_IDS_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# TRAIN_IDS_60=[1, 2]

TRAIN_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
             50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98,
             100, 103]
TRAIN_VALID_IDS = ([1, 2, 5, 8, 9, 13, 14, 15, 16, 18, 19, 27, 28, 31, 34, 38], [4, 17, 25, 35])
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')
SAMPLE_NUM = 2048
TRAIN_SET = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

class NTU_RGBD_new(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 full_train=True,
                 test=False,
                 validation=False,
                 DATA_CROSS_VIEW=True,
                 Transform=False,
                 DATA_CROSS_SET = False):

        self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
        self.root_path = root_path
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.transform = Transform
        self.depth_path = opt.depth_path


        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()
        self.TRAIN_IDS = TRAIN_IDS
        if opt.dataset == 'ntu60':
            indx = self.point_vids.index('S017C003P020R002A060.npy')# S004C001P003R001A001.npy# 'S002C001P003R001A001.mat' # ('S003C001P001R001A001.npy.mat')#('S017C003P020R002A060.npy')
            self.point_vids = self.point_vids[0:indx]
            self.TRAIN_IDS = TRAIN_IDS_60

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])
        self.opt = opt

        self.train = (test == False) and (validation == False)
        if DATA_CROSS_SET ==False:       
            if DATA_CROSS_VIEW == False:
                if test:
                    self.vid_ids = self.test_split_subject.copy()
                elif validation:
                    self.vid_ids = self.validation_split_subject.copy()
                elif full_train:
                    self.vid_ids = self.train_split_subject.copy()
                else:
                    self.vid_ids = self.train_split_subject_with_validation.copy()
            else:
                if test:
                    self.vid_ids = self.test_split_camera.copy()
                else:
                    self.vid_ids = self.train_split_camera.copy()
        else:
            if test:
                self.vid_ids = self.test_split_set.copy()
            else:
                self.vid_ids = self.train_split_set.copy()

        print('num_data:', len(self.vid_ids))


        self.point_clouds = np.empty(shape=[self.SAMPLE_NUM, self.INPUT_FEATURE_NUM], dtype=np.float32)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:20]


        if self.opt.branch_choose=='0':
         # motion training
            points_path = '../ntu/3DV_ntu60/reslution/Resolution60/raw/' + v_name + '.npy'
            key_path = '../ntu/3DV_ntu60/reslution/Resolution60/others/' + v_name + '_key.npy'
            resl_path  = '../ntu/3DV_ntu60/reslution/Resolution30/raw/' + v_name + '.npy'
            res2_path  = '../ntu/3DV_ntu60/reslution/Resolution10/raw/' + v_name + '.npy'
            points=np.load(points_path)
            res_points_1=np.load(resl_path)
            res_points_2 = np.load(res2_path)
            key_points = np.load(key_path)
            
            time_seg2 = self.get_temporal_augment_data(points, 4)
            time_seg4 = self.get_temporal_augment_data(points, 7)
            
            out_points = self.get_data_train(points[:, :4], key_points[:, :4], time_seg2[:, :4], time_seg4[:, :4],res_points_1[:, :4], res_points_2[:, :4], num_crop=10)
            label = self.id_to_action[vid_id]
            
            return out_points, v_name, label

        else:
            points_path = '../ntu/3DV_ntu60/reslution/Resolution60/app/' + v_name + '.npy'
            key_path = '../ntu/3DV_ntu60/reslution/Resolution60/others/' + v_name + '_key.npy'
            resl_path  = '../ntu/3DV_ntu60/reslution/Resolution30/app/' + v_name + '.npy'
            res2_path  = '../ntu/3DV_ntu60/reslution/Resolution10/app/' + v_name + '.npy'
            points=np.load(points_path)
            res_points_1=np.load(resl_path)
            res_points_2 = np.load(res2_path)
            key_points = np.load(key_path)
            
            time_seg2 = self.get_temporal_augment_data(points, 4)
            time_seg4 = self.get_temporal_augment_data(points, 7)
            
            out_points = self.get_data_train(points[:, :4], key_points[:, :4], time_seg2[:, :4], time_seg4[:, :4],res_points_1[:, :4], res_points_2[:, :4], num_crop=10)
            label = self.id_to_action[vid_id]
            
            return out_points, v_name, label
        
        













        # # print(v_name)
        # ex_feature = self.opt.ex_feature

        # if ex_feature == 0:
        #     key_points_path='../ntu/3DV_ntu60/others/'+v_name+'_key.npy'
        #     points_path = '../ntu/3DV_ntu60/raw/' + v_name + '.npy'
        
        #     points=np.load(points_path)
        #     key_points=np.load(key_points_path)
        #     time_seg2 = self.get_temporal_augment_data(points, 4)
        #     time_seg4 = self.get_temporal_augment_data(points, 7)
            
        #     out_points = self.deal_data_4(points[:, :4], key_points[:, :4], time_seg2, time_seg4, num_crop = 10)
        #     label = self.id_to_action[vid_id]
            
        #     return out_points, v_name, label
        # elif ex_feature==2:  # spatial ....
        #     key_points_path='../ntu/3DV_ntu60/others/'+v_name+'_key.npy'
        #     points_path = '../ntu/3DV_ntu60/raw/' + v_name + '.npy'
        
        #     points=np.load(points_path)
        #     key_points=np.load(key_points_path)
            
        #     out_points = self.deal_data_saptial(points[:, :4], key_points[:, :4], num_crop = 5)
        #     label = self.id_to_action[vid_id]
            
        #     return out_points, v_name, label
        
        # elif ex_feature==3:  # temporal ....
        #     # key_points_path='../ntu/3DV_ntu60/others/'+v_name+'_key.npy'
        #     points_path = '../ntu/3DV_ntu60/raw/' + v_name + '.npy'
        #     resl_path  = '../ntu/3DV_ntu60/reslution/Resolution30/raw/' + v_name + '.npy'
        #     res2_path  = '../ntu/3DV_ntu60/reslution/Resolution10/raw/' + v_name + '.npy'
        #     points=np.load(points_path)
        #     res_points=np.load(resl_path)
        #     res_points = np.load(res2_path)
            
        #     time_seg2 = self.get_temporal_augment_data(points, 4)
        #     time_seg4 = self.get_temporal_augment_data(points, 7)
            
        #     out_points = self.deal_data_temporal(points[:, :4], time_seg2[:, :4], time_seg4[:, :4],res_points[:, :4], res_points[:, :4], num_crop=5)
        #     label = self.id_to_action[vid_id]
            
        #     return out_points, v_name, label
        
        # elif ex_feature==4:

        #     vid_id = self.vid_ids[idx]
        #     vid_name = self.id_to_vidName[vid_id]
        #     v_name = vid_name[:20]
        #     base_path = '/data/data1/ntu/test_point_cloud/raw/'
        #     r_points = np.load(base_path + 'raw/' + v_name +'.npy')
        #     rs_points = np.load(base_path + 'res/' + v_name +'.npy')
        #     k_points = np.load(base_path + 'key/' + v_name +'.npy')
        #     t_points = np.load(base_path + 'time/' + v_name +'.npy')
        #     label = self.id_to_action[vid_id]
        #     out_points = self.deal_data_test( r_points[:, :, :4], rs_points[:, :, :4], k_points[:, :, :4], t_points[:, :, :4], num_crop = 10)
        #     return out_points, v_name, label
        
        # elif ex_feature==5:
    
        #     vid_id = self.vid_ids[idx]
        #     vid_name = self.id_to_vidName[vid_id]
        #     v_name = vid_name[:20]
        #     base_path = '/data/data1/ntu/test_point_cloud/app/'
        #     r_points = np.load(base_path + 'raw/' + v_name +'_app.npy')
        #     rs_points = np.load(base_path + 'res/' + v_name +'_app.npy')
        #     k_points = np.load(base_path + 'key/' + v_name +'_app.npy')
        #     t_points = np.load(base_path + 'time/' + v_name +'_app.npy')
        #     label = self.id_to_action[vid_id]
        #     out_points = self.deal_data_test_app( r_points[:, :, :4], rs_points[:, :, :4], k_points[:, :, :4], t_points[:, :, :4], num_crop = 10)
        #     return out_points, v_name, label
        
        # elif ex_feature==7:  
        #     # normal training
        #     points_path = '../ntu/3DV_ntu60/reslution/Resolution60/raw/' + v_name + '.npy'
        #     key_path = '../ntu/3DV_ntu60/reslution/Resolution60/others/' + v_name + '_key.npy'
        #     resl_path  = '../ntu/3DV_ntu60/reslution/Resolution30/raw/' + v_name + '.npy'
        #     res2_path  = '../ntu/3DV_ntu60/reslution/Resolution10/raw/' + v_name + '.npy'
        #     points=np.load(points_path)
        #     res_points_1=np.load(resl_path)
        #     res_points_2 = np.load(res2_path)
        #     key_points = np.load(key_path)
            
        #     time_seg2 = self.get_temporal_augment_data(points, 4)
        #     time_seg4 = self.get_temporal_augment_data(points, 7)
            
        #     out_points = self.get_data_train(points[:, :4], key_points[:, :4], time_seg2[:, :4], time_seg4[:, :4],res_points_1[:, :4], res_points_2[:, :4], num_crop=10)
        #     label = self.id_to_action[vid_id]
            
        #     return out_points, v_name, label
        
        # elif ex_feature==8:   # demonstreate less negative catorgry

        #     points_path = '../ntu/3DV_ntu60/raw/' + v_name + '.npy'
        #     key_path = '../ntu/3DV_ntu60/others/' + v_name + '_key.npy'
        #     resl_path  = '../ntu/3DV_ntu60/reslution/Resolution30/raw/' + v_name + '.npy'
        #     res2_path  = '../ntu/3DV_ntu60/reslution/Resolution10/raw/' + v_name + '.npy'
        #     points=np.load(points_path)
        #     res_points_1=np.load(resl_path)
        #     res_points_2 = np.load(res2_path)
        #     key_points = np.load(key_path)
            
        #     time_seg2 = self.get_temporal_augment_data(points, 4)
        #     time_seg4 = self.get_temporal_augment_data(points, 7)
            
        #     out_points = self.get_data_train_negative(points[:, :4], key_points[:, :4], time_seg2[:, :4], time_seg4[:, :4],res_points_1[:, :4], res_points_2[:, :4], num_crop=10)
        #     label = self.id_to_action[vid_id]
            
        #     return out_points, v_name, label
        
        # else:
        #     base_path = "../ntu/3DV_ntu60_fps_1020/"
        #     key_points_path = base_path + 'key/' + v_name+'_key.npy'
        #     points_path = base_path +'raw/' + v_name + '.npy'
        #     time_points_path1 = base_path +'time/' + v_name + '_seg4.npy'
        #     time_points_path2 = base_path +'time/' + v_name + '_seg5.npy'
        #     time_points_path3 = base_path +'time/' + v_name + '_seg6.npy'
        #     time_points_path4 = base_path +'time/' + v_name + '_seg7.npy'
        #     time_seg1 = np.load(time_points_path1)
        #     time_seg2 = np.load(time_points_path2)
        #     time_seg3 = np.load(time_points_path3)
        #     time_seg4 = np.load(time_points_path4)

        #     points=np.load(points_path)
        #     key_points=np.load(key_points_path)
        #     label = self.id_to_action[vid_id]
        #     return points[:,0:4], v_name, key_points_path, key_points[:,0:4], label, time_seg1[:,0:4], time_seg2, time_seg3,time_seg4
    
    def get_data_train(self, points, key_points, time_seg2, time_seg4,res_points_1, res_points_2, num_crop= 10):
        batch_size = 1
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        raw_p  = points[idex].copy().reshape(1, NUM_POINT, 4) 
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        rev_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        rev_p[:, :, :3] = self.jitter_point_cloud(rev_p[:, :, :3])
        rev_p = self.reverse_transform(rev_p)
        
        idex = np.random.randint(0, key_points.shape[0], NUM_POINT)
        ke1_p = key_points[idex].copy().reshape(1, NUM_POINT, 4) 
        ke1_p[:, :, :3] = self.jitter_point_cloud(ke1_p[:, :, :3])     
        idex = np.random.randint(0, key_points.shape[0], NUM_POINT)
        ke2_p = key_points[idex].copy().reshape(1, NUM_POINT, 4)
        ke2_p[:, :, :3] = self.jitter_point_cloud(ke2_p[:, :, :3])
        ke2_p = self.reverse_transform(ke2_p)
        
        
        
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro1_p = points[idex].copy().reshape(1, NUM_POINT, 4) 
        ro1_p[:, :, :3] = self.jitter_point_cloud(ro1_p[:, :, :3]) 
        ro1_p = self.rotate_trans(ro1_p)
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro2_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        ro2_p[:, :, :3] = self.jitter_point_cloud(ro2_p[:, :, :3]) 
        ro2_p = self.rotate_trans(ro2_p)       
   
        ti1_p = time_seg2.reshape(1, NUM_POINT, 4)      
        ti2_p = time_seg4.reshape(1, NUM_POINT, 4)
        
        idex = np.random.randint(0, res_points_1.shape[0], NUM_POINT)
        rs1_p = res_points_1[idex].copy().reshape(1, NUM_POINT, 4)       
        idex = np.random.randint(0, res_points_2.shape[0], NUM_POINT)
        rs2_p = res_points_2[idex].copy().reshape(1, NUM_POINT, 4)
        

        
        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  raw_p
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  rev_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke2_p    
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs2_p
        crop_i+=1
        return data_pairs 
    
    def get_data_train_negative(self, points, key_points, time_seg2, time_seg4,res_points_1, res_points_2, num_crop= 10):
        batch_size = 1
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        raw_p  = points[idex].copy().reshape(1, NUM_POINT, 4) 
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        rev_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        rev_p[:, :, :3] = self.jitter_point_cloud(rev_p[:, :, :3])
        
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro1_p = points[idex].copy().reshape(1, NUM_POINT, 4) 
        ro1_p[:, :, :3] = self.jitter_point_cloud(ro1_p[:, :, :3]) 
        ro1_p = self.rotate_trans(ro1_p)
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro2_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        ro2_p[:, :, :3] = self.jitter_point_cloud(ro2_p[:, :, :3]) 
        ro2_p = self.rotate_trans(ro2_p)
        
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro3_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        ro3_p[:, :, :3] = self.jitter_point_cloud(ro3_p[:, :, :3]) 
        ro3_p = self.rotate_trans(ro3_p)   
        
        idex = np.random.randint(0, points.shape[0], NUM_POINT)
        ro4_p = points[idex].copy().reshape(1, NUM_POINT, 4)
        ro4_p[:, :, :3] = self.jitter_point_cloud(ro4_p[:, :, :3]) 
        ro4_p = self.rotate_trans(ro4_p)    
        
        idex = np.random.randint(0, res_points_1.shape[0], NUM_POINT)
        rs1_p = res_points_1[idex].copy().reshape(1, NUM_POINT, 4)       
        idex = np.random.randint(0, res_points_2.shape[0], NUM_POINT)
        rs2_p = res_points_2[idex].copy().reshape(1, NUM_POINT, 4)
        
        idex = np.random.randint(0, res_points_1.shape[0], NUM_POINT)
        rs3_p = res_points_1[idex].copy().reshape(1, NUM_POINT, 4)       
        idex = np.random.randint(0, res_points_2.shape[0], NUM_POINT)
        rs4_p = res_points_2[idex].copy().reshape(1, NUM_POINT, 4)
        

        
        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  raw_p
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  rev_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro3_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro4_p    
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs3_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs4_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs2_p
        crop_i+=1
        return data_pairs 
        
        
        
    
    def deal_data_test(self, r_points, rs_points, k_points, t_points, num_crop =10): 
        batch_size = 1
        raw_p = r_points[0:1]
        ro1_p = r_points[1:2]
        ro2_p = r_points[2:3]
        rev_p = r_points[3:4]
        
        ti1_p = t_points[0:1]
        ti2_p = t_points[3:4]
        
        rs1_p = rs_points[0:1]
        rs2_p = rs_points[1:2]
        
        ke1_p = k_points[1:2]
        ke2_p = k_points[0:1]
        
        ke2_p = self.reverse_transform(ke2_p)
        rev_p = self.reverse_transform(rev_p)
        ro1_p = self.depth_transform(ro1_p, -1)
        ro2_p = self.depth_transform(ro2_p, 1)


        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  raw_p
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  rev_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke2_p    
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs2_p
        crop_i+=1
        return data_pairs 
    
    def deal_data_test_app(self, r_points, rs_points, k_points, t_points, num_crop =10): 
        batch_size = 1
        raw_p = r_points[0:1]
        ro1_p = r_points[1:2]
        ro2_p = r_points[2:3]
        rev_p = r_points[3:4]
        
        ti1_p = t_points[0:1]
        ti2_p = t_points[1:2]
        
        rs1_p = rs_points[0:1]
        rs2_p = rs_points[1:2]
        
        ke1_p = k_points[1:2]
        ke2_p = k_points[0:1]
        
        ke2_p = self.reverse_transform(ke2_p)
        rev_p = self.reverse_transform(rev_p)
        ro1_p = self.depth_transform(ro1_p, -1)
        ro2_p = self.depth_transform(ro2_p, 1)


        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  raw_p
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  rev_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ke2_p    
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ti2_p
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs1_p
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rs2_p
        crop_i+=1
        return data_pairs 
    
    
    
    def deal_data_saptial(self, points, key_point, num_crop = 5): 
        batch_size = 1
        N, C = points.shape
        points = points.reshape(1, N,C)
        key_point = key_point.reshape(1, N,C)
        scale_data = np.concatenate((key_point.copy(), points.copy()), 1)
        points, key_point = self.points_sample_jiter(points, key_point)
        points = self.fps_sample_data(points, sample_num_level1, sample_num_level2)
        points_2 = self.reverse_transform(points)
        
        key_point = self.fps_sample_data(key_point, sample_num_level1, sample_num_level2)

        idex = np.random.randint(0, scale_data.shape[1], NUM_POINT)
        scale_data = scale_data[:, idex, :]
        
        rotate_data = self.rotate_trans(scale_data)
        scale_data = self.scale_trans(points)

        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  points
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  points_2
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  key_point
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  scale_data    
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  rotate_data
        crop_i+=1

        return data_pairs
    
    def deal_data_temporal(self, points, time1, time2, res1, res2, num_crop = 5): 
        batch_size = 1
        N, C = points.shape
        points = points.reshape(1, N,C)
        time1 = time1.reshape(1, -1,4)
        time2 = time2.reshape(1, -1,4)
        res1 = res1.reshape(1, -1,C)
        res2 = res2.reshape(1, -1,C)
        
        
        points, time1 = self.points_sample_jiter(points, time1)
        points = self.fps_sample_data(points, sample_num_level1, sample_num_level2)
        
        time1 = self.fps_sample_data(time1, sample_num_level1, sample_num_level2)

        idex = np.random.randint(0, time2.shape[1], NUM_POINT)
        time2 = time2[:, idex, :]
        idex = np.random.randint(0, res1.shape[1], NUM_POINT)
        res1 = res1[:, idex, :]
        idex = np.random.randint(0, res2.shape[1], NUM_POINT)
        res2 = res2[:, idex, :]
        
        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  points
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  time1
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  time2
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  res1    
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  res2
        crop_i+=1

        return data_pairs

    def deal_data_4(self, points, key_point, time_seg2, time_seg4, num_crop): 
        batch_size = 1
        N, C = points.shape
        points = points.reshape(1, N,C)
        key_point = key_point.reshape(1, N,C)

        points, key_point = self.points_sample_jiter(points, key_point)

        points = self.fps_sample_data(points, sample_num_level1, sample_num_level2)
        points_2 = self.reverse_transform(points)
        
        key_point = self.fps_sample_data(key_point, sample_num_level1, sample_num_level2)
        key_point_2 = self.reverse_transform(key_point)

        deep_data = self.depth_transform(points, -1)
        deep_data_2 = self.depth_transform(points, 1)

        scale_data = self.rank_transform(points, rank_slop=0.6)
        scale_data_2 = self.rank_transform(points, rank_slop=1.4)

        time_data = time_seg2.reshape(1, NUM_POINT, 4)
        time_data_2 = time_seg4.reshape(1, NUM_POINT, 4)

        data_pairs = np.empty([num_crop, NUM_POINT, 4], dtype=float)

        crop_i = 0
        data_pairs[crop_i :(crop_i+1) , :, :] =  points
        crop_i+=1
        data_pairs[crop_i :(crop_i+1) , :, :] =  points_2
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  key_point
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  key_point_2    
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  deep_data
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  deep_data_2
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  scale_data
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  scale_data_2
        crop_i+=1

        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  time_data
        crop_i+=1
        data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  time_data_2
        crop_i+=1
        # data_pairs[:, :, 0] = 0
        return data_pairs
    
    def get_temporal_augment_data(self, pointss, temporal_int):
        points = pointss.copy()
        point_temporal = np.concatenate((points[:, 0:3].copy(), points[:, temporal_int:temporal_int+1].copy()), axis = 1)
        idexx = np.where(point_temporal[:, 3]!=0)
        point_temporal = point_temporal[idexx]
        idex = np.random.randint(0, point_temporal.shape[0], 512)
        point_temporal = point_temporal[idex]
        # point_temporal = self.jitter_point_cloud(point_temporal)
        # point_temporal = self.get_fps_data(point_temporal)
        return point_temporal
    
    def fps_sample_data(self, points_xyzc, sample_num_level1, sample_num_level2):
        # NUM_POINT = 512
        for kk in range(points_xyzc.shape[0]):
            sampled_idx_l1 = self.farthest_point_sampling_fast(points_xyzc[kk, :, 0:3], sample_num_level1)
            other_idx = np.setdiff1d(np.arange(NUM_POINT), sampled_idx_l1.ravel())
            new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
            points_xyzc[kk, :, :] = points_xyzc[kk, new_idx[:NUM_POINT], :]
        return points_xyzc


    def farthest_point_sampling_fast(self, pc, sample_num):
        pc_num = pc.shape[0]

        sample_idx = np.zeros(shape=[sample_num, 1], dtype=np.int32)
        sample_idx[0] = np.random.randint(0, pc_num)

        cur_sample = np.tile(pc[sample_idx[0], :], (pc_num, 1))
        diff = pc - cur_sample
        min_dist = (diff * diff).sum(axis=1)

        for cur_sample_idx in range(1, sample_num):
            ## find the farthest point

            sample_idx[cur_sample_idx] = np.argmax(min_dist)
            if cur_sample_idx < sample_num - 1:
                diff = pc - np.tile(pc[sample_idx[cur_sample_idx], :], (pc_num, 1))
                min_dist = np.concatenate((min_dist.reshape(pc_num, 1), (diff * diff).sum(axis=1).reshape(pc_num, 1)),
                                        axis=1).min(axis=1)  ##?
        # print(min_dist)
        return sample_idx


    def points_sample_jiter(self, points, key_point):
        idex = np.random.randint(0, points.shape[1], size=NUM_POINT)
        points = points[:, idex, :]
        idex = np.random.randint(0, key_point.shape[1], size=NUM_POINT)
        key_point = key_point[:, idex, :]
        key_point[:, :, 0:3] = self.jitter_point_cloud(key_point[:, :, 0:3])
        points[:, :, 0:3] = self.jitter_point_cloud(points[:, :, 0:3])
        return points, key_point



    def reverse_transform(self, points):
        reverse_data = np.zeros(points.shape, dtype=np.float32)
        reverse_data[:, :, :] = points[:, :, :]
        reverse_data[:, :, 0] = -reverse_data[:, :, 0]
        reverse_data[:, :, 0:3] = self.jitter_point_cloud(reverse_data[:, :, 0:3])
        return reverse_data


    def depth_transform(self, points,  angle_set):
        rotated_data = np.zeros(points.shape, dtype=np.float32)
        rotated_data[:, :, :] = points[:, :, :]
        for k in range(rotated_data.shape[0]):
            # depth transform
            # rotate

            angle =  angle_set * np.pi* 0.25
            Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
            shape_pc = rotated_data[k, :, 0:3]
            # print(shape_pc.shape)
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), Ry)
        # normalize
        # nor_data = normalize_data(rotated_data)
        return rotated_data
    
    def rotate_trans(self, points):
        rotated_data = np.zeros(points.shape, dtype=np.float32)
        rotated_data[:, :, :] = points[:, :, :]
        for k in range(rotated_data.shape[0]):

            angle =  (np.random.rand()-0.5) * np.pi * 0.8
            Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
            shape_pc = rotated_data[k, :, 0:3]
            # print(shape_pc.shape)
            rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), Ry)
        # normalize
        # nor_data = normalize_data(rotated_data)
        return rotated_data
    
    def rank_transform(self, points, rank_slop=-1):
        rank_data = np.zeros(points.shape, dtype=np.float32)
        rank_data[:, :, :] = points[:, :, :]
        rank_data = rank_slop * rank_data
        # rank_data=rank_data-rank_data.min()
        return rank_data
    
    def scale_trans(self, points):
        
        rank_slop = np.random.rand()+0.5
        rank_data = points.copy()
        rank_data[:, :, :3] = rank_slop *  rank_data[:, :, :3]
        # rank_data=rank_data-rank_data.min()
        return rank_data

    def jitter_point_cloud(self, batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data
    
    

    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):
    
        vid_name = self.point_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            'video_set': (setup, camera),
            'setup': setup,
            'camera': camera,
            'performer': performer,
            'replication': replication,
            'action': action,
        }
        

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        train_ids_camera = [2, 3]

        # Cross-Subject splits
        self.train_split_subject = list(
            dataset[dataset.performer.isin(self.TRAIN_IDS)]['video_index'])
        self.train_split_subject_with_validation = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[0])]['video_index'])
        self.validation_split_subject = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[1])]['video_index'])
        self.test_split_subject = list(
            dataset[~dataset.performer.isin(self.TRAIN_IDS)]['video_index'])

        # Cross-View splits
        self.train_split_camera = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera = list(
            dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])
        
        # Cross Set splits
        self.train_split_set = list(
            dataset[dataset.setup.isin(TRAIN_SET)]['video_index'])
        self.test_split_set = list(
            dataset[~dataset.setup.isin(TRAIN_SET)]['video_index'])




class NTU_RGBD(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 full_train=True,
                 test=False,
                 validation=False,
                 DATA_CROSS_VIEW=True,
                 Transform=False):

        self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
        self.root_path = root_path
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.transform = Transform
        self.depth_path = opt.depth_path


        self.point_vids = os.listdir(self.root_path)
        self.point_vids.sort()

        self.TRAIN_IDS = TRAIN_IDS
        if opt.dataset == 'ntu60':
            indx = self.point_vids.index('S017C003P020R002A060.npy')# S004C001P003R001A001.npy# 'S002C001P003R001A001.mat' # ('S003C001P001R001A001.npy.mat')#('S017C003P020R002A060.npy')
            self.point_vids = self.point_vids[0:indx]
            self.TRAIN_IDS = TRAIN_IDS_60

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.point_data = self.load_data()
        self.set_splits()
        self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
        self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

        self.train = (test == False) and (validation == False)
        if DATA_CROSS_VIEW == False:
            if test:
                self.vid_ids = self.test_split_subject.copy()
            elif validation:
                self.vid_ids = self.validation_split_subject.copy()
            elif full_train:
                self.vid_ids = self.train_split_subject.copy()
            else:
                self.vid_ids = self.train_split_subject_with_validation.copy()
        else:
            if test:
                self.vid_ids = self.test_split_camera.copy()
            else:
                self.vid_ids = self.train_split_camera.copy()

        print('num_data:', len(self.vid_ids))

        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM

        self.point_clouds = np.empty(shape=[self.SAMPLE_NUM, self.INPUT_FEATURE_NUM], dtype=np.float32)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        vid_name = self.id_to_vidName[vid_id]
        v_name = vid_name[:20] 
        # print(v_name)
        key_points_path='../ntu/3DV_ntu60/others/'+v_name+'_key.npy'
        points_path = '../ntu/3DV_ntu60/raw/' + v_name + '.npy'
        # att_path = '../ntu/ntu60_new2/att/' + v_name + '_att.npy'

        # time seg part:
        time_points_path1 = '../ntu/3DV_ntu60/reslution/Resolution30/raw/' + v_name + '.npy'
        time_points_path2 = '../ntu/3DV_ntu60/timeseg_2048/' + v_name + '_seg5.npy'
        time_points_path3 = '../ntu/3DV_ntu60/timeseg_2048/' + v_name + '_seg6.npy'
        time_points_path4 = '../ntu/3DV_ntu60/timeseg_2048/' + v_name + '_seg7.npy'

        time_seg1 = np.load(time_points_path1)
        time_seg2 = np.load(time_points_path2)
        time_seg3 = np.load(time_points_path3)
        time_seg4 = np.load(time_points_path4)

        points=np.load(points_path)
        key_points=np.load(key_points_path)
        label = self.id_to_action[vid_id]
        # points1 = np.hstack((points[:, 0:4], points[:, 5:6]))
        # key_points1 = np.hstack((key_points[:, 0:4], key_points[:, 5:6]))
        idx = np.random.randint(0, time_seg1.shape[0], 512)
        time_seg1 = time_seg1[idx]
        return points[:,0:4], v_name, key_points_path, key_points[:,0:4], label, time_seg1[:,0:4], time_seg2, time_seg3,time_seg4
        # return points, points_path, key_points_path, key_points, label


    def __len__(self):
        return len(self.vid_ids)


    def load_data(self):
        self.point_data = []
        for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
            self.point_data.append(self.get_pointdata(cloud_idx))

        return self.point_data

    def get_pointdata(self, vid_id):

        vid_name = self.point_vids[vid_id]
        match = re.match(compiled_regex, vid_name)
        setup, camera, performer, replication, action = [*map(int, match.groups())]
        return {
            'video_cloud_name': vid_name,
            'video_index': vid_id,
            'video_set': (setup, camera),
            'setup': setup,
            'camera': camera,
            'performer': performer,
            'replication': replication,
            'action': action,
        }

    def set_splits(self):
        '''
        Sets the train/test splits
        Cross-Subject Evaluation:
            Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
                        28, 31, 34, 35, 38
        Cross-View Evaluation:
            Train camera views: 2, 3
        '''
        # Save the dataset as a dataframe
        dataset = pd.DataFrame(self.point_data)

        # Get the train split ids
        train_ids_camera = [2, 3]

        # Cross-Subject splits
        self.train_split_subject = list(
            dataset[dataset.performer.isin(self.TRAIN_IDS)]['video_index'])
        self.train_split_subject_with_validation = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[0])]['video_index'])
        self.validation_split_subject = list(
            dataset[dataset.performer.isin(TRAIN_VALID_IDS[1])]['video_index'])
        self.test_split_subject = list(
            dataset[~dataset.performer.isin(self.TRAIN_IDS)]['video_index'])

        # Cross-View splits
        self.train_split_camera = list(
            dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
        self.test_split_camera = list(
            dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])
