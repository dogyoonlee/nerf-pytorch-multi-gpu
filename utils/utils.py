import numpy as np
import torch
import os
from utils.metrics import compute_img_metric
import mediapy as media
from matplotlib import cm
from PIL import Image

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

def compute_all_metrics(pred, target, metric_list):
    test_val = dict()
    for met in metric_list:
        test_val[met] = compute_img_metric(pred, target, met)
    return test_val

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

def open_file(pth, mode='r'):
    return open(pth, mode=mode)

def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open_file(pth, 'wb') as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(f, 'PNG')

def save_img(img, pth, vis_type=None):
    """Save an image in [0, 255] to disk as a PNG."""
    with open_file(pth, 'wb') as f:
        if vis_type is not None and vis_type != 'gray': # for depth image and depth_img_vis_type.
            img = img / 255.
            img = cm.get_cmap(vis_type)(img)[..., :3]
            img = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        Image.fromarray(img.astype(np.uint8)).save(f, 'PNG')

def create_videos(imgs, save_dir, prefix='', render_data_type='color', render_dist_percentile=0.5, 
                  render_video_fps=30, render_video_crf=18, render_dist_curve_fn=np.log, depth_vis_type='gray'):
    """Creates videos out of the images saved to disk."""
    video_file = os.path.join(save_dir, f'{prefix}{render_data_type}_video.mp4')
    
    os.makedirs(save_dir, exist_ok=True)
    
    shape = imgs.shape
    num_frames = imgs.shape[0]
    print(f'Video shape is {shape[1:3]}')
    
    if render_data_type.startswith('depth_'):
        p = render_dist_percentile
        depth_limits = np.percentile(imgs.flatten(), [p, 100 - p])
        lo, hi = [render_dist_curve_fn(x) for x in depth_limits]

    video_kwargs = {
        'shape': shape[1:3], # shape: [num_frame x H x W x 3]
        'codec': 'h264',
        'fps': render_video_fps,
        'crf': render_video_crf,
    }
    
    for k in ['color', 'normals', 'acc', 'depth', 'depth_mean', 'depth_median']:
        video_file = os.path.join(save_dir, f'{prefix}{k}_video.mp4')
        if k == 'acc':
            input_format = 'gray'
        elif k.startswith('depth') and depth_vis_type == 'gray':
            input_format = 'gray'
        else:
            input_format = 'rgb'
            
        if k != render_data_type:
            continue
        
        print(f'Making video {video_file}...')
        with media.VideoWriter(
            video_file, **video_kwargs, input_format=input_format) as writer:
            for idx in range(num_frames):
                img = imgs[idx]
                if k in ['color', 'depth', 'normals']:
                    img = img / 255.
                elif k.startswith('depth_'):
                    img = render_dist_curve_fn(img)
                    img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                
                if k.startswith('depth') and depth_vis_type != 'gray':
                    img = cm.get_cmap(depth_vis_type)(img)[..., :3]

                frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
                writer.add_image(frame)
                
