import os
import sys
from matplotlib.pyplot import axis
import numpy as np
import math
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

NUM_POINT = 512
sample_num_level1 = 64
sample_num_level2 = 64
ntu10 = [1, 8, 10, 14, 23, 24, 35, 52, 58, 59]

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
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




def points_sample_jiter(points, key_point):
    idex = np.random.randint(0, points.shape[1], size=NUM_POINT)
    points = points[:, idex, :]
    idex = np.random.randint(0, key_point.shape[1], size=NUM_POINT)
    key_point = key_point[:, idex, :]
    key_point[:, :, 0:3] = jitter_point_cloud(key_point[:, :, 0:3])
    return points, key_point



def deal_data_simclr(points, key_point,time_seg1, time_seg2, time_seg3,time_seg4, num_crop):
    points = points.numpy()
    key_point = key_point.numpy()
    batch_size = points.shape[0]
    time_seg2 = time_seg2.numpy()
    time_seg4 = time_seg4.numpy()
    idex = np.random.randint(0, time_seg4.shape[1], size=NUM_POINT)
    time_seg2 = time_seg2[:, idex, :]
    time_seg4 = time_seg4[:, idex, :]   
    points, key_point = points_sample_jiter(points, key_point)

    points = fps_sample_data(points, sample_num_level1, sample_num_level2)
    key_point = fps_sample_data(key_point, sample_num_level1, sample_num_level2)

    data_raw_pairs = np.empty([4, batch_size, NUM_POINT, 4], dtype=float)
    data_raw_pairs[0] = points.copy()
    data_raw_pairs[1] = key_point.copy()
    data_raw_pairs[2] = time_seg2.copy()
    data_raw_pairs[3] = time_seg4.copy()

    base_data_idex = np.random.randint(0, 4,(2))
    
    data_1 = data_raw_pairs[base_data_idex[0]].copy()
    data_2 = data_raw_pairs[base_data_idex[1]].copy()

    data_1 = get_random_augment(data_1)
    data_2 = get_random_augment(data_2)


    data_pairs = np.empty([num_crop * batch_size, NUM_POINT, 4], dtype=float)
    crop_i = 0
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  data_1
    crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  data_2
    crop_i+=1

    return data_pairs

def deal_simclr_new(points_r, points_temp, ratio =0.5):   
    A, B, N, D = points_r.shape
    points_r = points_r.reshape(-1, 2048, 4)
    # points_temp = points_temp.reshape(-1, 2048, 4)
    idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    points = points_r[:, idex, :].copy()
    idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    # points_temp = points_r[:, idex, :].copy()
    points_temp = points.copy()
    
    data_1 = get_random_augment(points)
    data_2 = get_random_augment(points_temp)
    data_1 = data_1.reshape(int(A*ratio),int(1/ratio), int(1/ratio),int(B*ratio), NUM_POINT, D).transpose(0, 2, 1, 3, 4, 5).reshape(A, B, NUM_POINT, D)
    data_2 = data_2.reshape(int(A*ratio),int(1/ratio), int(1/ratio),int(B*ratio), NUM_POINT, D).transpose(0, 2, 1, 3, 4, 5).reshape(A, B, NUM_POINT, D)
    ord_idex = np.arange(0, B, 1)  # 0~9
    # np.random.shuffle(ord_idex)
    # data_1 = data_1[:, ord_idex, :, :]
    # data_2 = data_2[:, ord_idex, :, :]
    data_pairs = np.concatenate((data_1, data_2), axis = 1)
    return data_pairs

def deal_simclr_new_test(points_r, points_temp, ratio =0.5):   
    points_r = points_r.numpy()
    # points_temp = points_temp.numpy()
    B, N, D = points_r.shape # batch * dim * D
    batch_size = B
    points_r = points_r.reshape(-1, 2048, 4)
    # points_temp = points_temp.reshape(-1, 2048, 4)
    idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    points = points_r[:, idex, :].copy()
    # points_temp = points_temp[:, idex, :]
    
    # idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    # ro1_p = points_r[:, idex, :].copy()
    # idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    # ro2_p = points_r[:, idex, :].copy()
    ro1_p = depth_transform(points.copy(), 1)
    ro2_p = depth_transform(points.copy(), -1)

    # ro3_p = depth_transform(points_temp.copy(), 1)
    # ro4_p = depth_transform(points_temp.copy(), -1)

    # sa1_p = rank_transform(points.copy(), 0.7)
    # sa2_p = rank_transform(points_temp.copy(), 0.7)
    
    # idex = np.random.randint(0, points_r.shape[1], size=NUM_POINT)
    # re1_p = points_r[:, idex, :].copy()
    re1_p = reverse_transform(points.copy())
    # re2_p = reverse_transform(points_temp.copy())

    data_pairs = np.empty([4 * B, NUM_POINT, 4], dtype=float)
    crop_i = 0
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points
    crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points_temp
    # crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro1_p
    crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro2_p
    crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro3_p
    # crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  ro4_p
    # crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  sa1_p
    # crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  sa2_p
    # crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  re1_p
    crop_i+=1
    # data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  re2_p
    # crop_i+=1

    return data_pairs


def get_random_augment(points):
    scale = np.random.rand()*0.8+ 0.6
    rotate = np.random.rand()*3-1.5
    reverse = np.random.randint(0, 2)
    temp_points = points.copy()
    temp_points = depth_transform(temp_points, rotate)
    temp_points = rank_transform(temp_points, rank_slop=scale)
    # print(scale, reverse, rotate)
    if reverse == 1:
        temp_points = reverse_transform(temp_points)
    
    temp_points = jitter_point_cloud(temp_points)
    return temp_points


#no fps >>>>>>>>>>>>>>>>>>>
def deal_data_4(points, key_point,time_seg1, time_seg2, time_seg3,time_seg4, num_crop):
    points = points.numpy()
    key_point = key_point.numpy()
    batch_size = points.shape[0]
    time_seg2 = time_seg2.numpy()
    time_seg4 = time_seg4.numpy()
    idex = np.random.randint(0, time_seg4.shape[1], size=NUM_POINT)
    time_seg2 = time_seg2[:, idex, :]
    time_seg4 = time_seg4[:, idex, :]   

    # key_point=rotate_point_cloud(points)
    points, key_point = points_sample_jiter(points, key_point)

    points = fps_sample_data(points, sample_num_level1, sample_num_level2)
    points_2 = reverse_transform(points)
    
    key_point = fps_sample_data(key_point, sample_num_level1, sample_num_level2)
    key_point_2 = reverse_transform(key_point)

    deep_data = depth_transform(points, -1)
    deep_data_2 = depth_transform(points, 1)

    scale_data = rank_transform(points, rank_slop=0.6)
    scale_data_2 = rank_transform(points, rank_slop=1.4)

    time_data = time_seg2
    time_data_2 = time_seg4

    data_pairs = np.empty([num_crop * batch_size, NUM_POINT, 4], dtype=float)

    crop_i = 0
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points
    crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points_2
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


# fps.....................
def deal_data_4_f(points, key_point,time_seg1, time_seg2, time_seg3,time_seg4, num_crop):
    points = points.numpy()
    key_point = key_point.numpy()
    batch_size = points.shape[0]
    time_seg2 = time_seg2.numpy()
    time_seg4 = time_seg4.numpy()

    key_point[:, :, 0:3] = jitter_point_cloud(key_point[:, :, 0:3])
    points[:, :, 0:3] = jitter_point_cloud(points[:, :, 0:3])

    points_2 = reverse_transform(points)
    
    key_point_2 = reverse_transform(key_point)

    deep_data = depth_transform(points, -1)
    deep_data_2 = depth_transform(points, 1)

    scale_data = rank_transform(points, rank_slop=0.6)
    scale_data_2 = rank_transform(points, rank_slop=1.4)

    time_data = time_seg2
    time_data_2 = time_seg4

    data_pairs = np.empty([num_crop * batch_size, NUM_POINT, 4], dtype=float)

    crop_i = 0
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points
    crop_i+=1
    data_pairs[crop_i * batch_size:(crop_i+1) * batch_size, :, :] =  points_2
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



def fps_sample_data(points_xyzc, sample_num_level1, sample_num_level2):
    for kk in range(points_xyzc.shape[0]):
        sampled_idx_l1 = farthest_point_sampling_fast(points_xyzc[kk, :, 0:3], sample_num_level1)
        other_idx = np.setdiff1d(np.arange(NUM_POINT), sampled_idx_l1.ravel())
        new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
        points_xyzc[kk, :, :] = points_xyzc[kk, new_idx, :]
        # 2nd
        sampled_idx_l2 = farthest_point_sampling_fast(points_xyzc[kk, 0:sample_num_level1, 0:3], sample_num_level2)
        other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
        new_idx = np.concatenate((sampled_idx_l2.ravel(), other_idx))
        points_xyzc[kk, 0:sample_num_level1, :] = points_xyzc[kk, new_idx, :]
    return points_xyzc


def farthest_point_sampling_fast(pc, sample_num):
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




def reverse_transform(points):
    reverse_data = np.zeros(points.shape, dtype=np.float32)
    reverse_data[:, :, :] = points[:, :, :]
    reverse_data[:, :, 0] = -reverse_data[:, :, 0]
    reverse_data[:, :, 0:3] = jitter_point_cloud(reverse_data[:, :, 0:3])
    return reverse_data


def depth_transform(points,  angle_set):
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


def normalize_data(points_xyzc):
    ## Normalization
    for i in range(points_xyzc.shape[0]):
        y_len = points_xyzc[i, :, 1].max() - points_xyzc[i, :, 1].min()
        x_len = points_xyzc[i, :, 0].max() - points_xyzc[i, :, 0].min()
        z_len = points_xyzc[i, :, 2].max() - points_xyzc[i, :, 2].min()
        c_max, c_min = points_xyzc[i, :, 3:].max(axis=0), points_xyzc[i, :, 3:].min(axis=0)
        c_len = c_max - c_min

        x_center = (points_xyzc[i, :, 0].max() + points_xyzc[i, :, 0].min()) / 2
        y_center = (points_xyzc[i, :, 1].max() + points_xyzc[i, :, 1].min()) / 2
        z_center = (points_xyzc[i, :, 2].max() + points_xyzc[i, :, 2].min()) / 2
        centers = np.tile([x_center, y_center, z_center], (points_xyzc.shape[1], 1))
        points_xyzc[i, :, 0:3] = (points_xyzc[i, :, 0:3] - centers) / y_len
    return points_xyzc


def rank_transform(points, rank_slop=-1):
    # rank_data = np.zeros(points.shape, dtype=np.float32)
    rank_data = points.copy()
    rank_data[:, :, :3] = rank_data[:, :, :3] * rank_slop
    # rank_data=rank_data-rank_data.min()
    return rank_data


def real_rank_trans(points, rank_slop=0.5):
    rank_slop = 0.7 * np.random.random() + 0.2
    rank_data = np.zeros(points.shape, dtype=np.float32)
    rank_data[:, :, :] = points[:, :, :]
    rank_data[:, :, 3:] = rank_slop * rank_data[:, :, 3:]
    # rank_data[:, :, 3] = rank_data[:, :, 3] - rank_data[:, :, 3].min()
    return rank_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
      Input:
        data: B,N,... numpy array
        label: B,... numpy array
      Return:
        shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx, ...], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data



def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data




