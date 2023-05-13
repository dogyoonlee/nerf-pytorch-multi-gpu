import numpy as np
import torch
from utils.metrics import compute_img_metric

def compute_time(dt):
    # train_time = time.time()-start_time
    dt_h = dt//3600
    dt_m = (dt - dt_h*3600)//60
    dt_s = dt - dt_h*3600 - dt_m*60
    return dt_h, dt_m, dt_s

def exponential_scale_fine_loss_weight(N_iters, kernel_start_iter, start_ratio, end_ratio, iter):
    interval_len = N_iters - kernel_start_iter
    scale = (1 / interval_len) * np.log(end_ratio / start_ratio)
    return start_ratio * np.exp(scale * (iter - kernel_start_iter))

def make_poses_with_dummy(poses, num_gpu=1):
    dummy_num = ((len(poses) - 1) // num_gpu + 1) * num_gpu - len(poses)
    # dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(poses)
    dummy_poses = torch.eye(poses.shape[-2], poses.shape[-2]).unsqueeze(0).expand(dummy_num, poses.shape[-2], poses.shape[-2]).type_as(poses)
    if dummy_num > 0:
        poses_w_dummy = torch.cat([poses, dummy_poses], dim=0)
    else:
        poses_w_dummy = poses
    
    print(f"Append {dummy_num} # of poses to fill all the GPUs")
    
    return poses_w_dummy, dummy_num

def compute_all_metrics(pred, target, metric_list):
    test_val = dict()
    for met in metric_list:
        test_val[met] = compute_img_metric(pred, target, met)
    return test_val