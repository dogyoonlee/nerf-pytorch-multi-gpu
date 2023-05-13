import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_time_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    all_times = []
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            times.append(int(frame['file_path'].split('/')[-1]))
            
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split, times




def load_blender_LensLess_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_depth_imgs = []
    all_normal_imgs = []
    all_ll_imgs = []
    all_poses = []
    counts = [0]
    not_img_counts = [0]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depth_imgs = []
        normal_imgs = []
        ll_imgs = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            
            if s != 'val':
                fname_depth = os.path.join(basedir, frame['file_path'] + '_depth_0186.png')
                fname_normal = os.path.join(basedir, frame['file_path'] + '_normal_0186.png')
                fname_ll = os.path.join(basedir, frame['file_path'] + '_raw.png')
                depth_imgs.append(imageio.imread(fname_depth))
                normal_imgs.append(imageio.imread(fname_normal))
                ll_imgs.append(imageio.imread(fname_ll))
                
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        depth_imgs = (np.array(depth_imgs)).astype(np.float32)
        normal_imgs = (np.array(normal_imgs) / 255.).astype(np.float32)
        ll_imgs = (np.array(ll_imgs) / 255.).astype(np.float32)
            
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)   
        all_poses.append(poses)
        
        all_depth_imgs.append(depth_imgs)
        all_normal_imgs.append(normal_imgs)
        all_ll_imgs.append(ll_imgs)
                    
        not_img_counts.append(not_img_counts[-1] + depth_imgs.shape[0])
        
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    i_split_not_imgs = [np.arange(not_img_counts[i], not_img_counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    if i_split_not_imgs[1].shape[0] == 0:
        all_depth_imgs.pop(1)
        all_normal_imgs.pop(1)
        all_ll_imgs.pop(1)
        
    depth_imgs = np.concatenate(all_depth_imgs, 0)
    normal_imgs = np.concatenate(all_normal_imgs, 0)
    ll_imgs = np.concatenate(all_ll_imgs, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        
        depth_imgs_half_res = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, depth_img_ in enumerate(depth_imgs):
            depth_imgs_half_res[i] = cv2.resize(depth_img_, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = depth_imgs_half_res
        
        normal_imgs_half_res = np.zeros((normal_imgs.shape[0], H, W, 4))
        for i, normal_img_ in enumerate(normal_imgs):
            normal_imgs_half_res[i] = cv2.resize(normal_img_, (W, H), interpolation=cv2.INTER_AREA)
        normal_imgs = normal_imgs_half_res
        
        ll_imgs_half_res = np.zeros((ll_imgs.shape[0], H, W, 4))
        for i, ll_img_ in enumerate(ll_imgs):
            ll_imgs_half_res[i] = cv2.resize(ll_img_, (W, H), interpolation=cv2.INTER_AREA)
        ll_imgs = ll_imgs_half_res

    
    return imgs, depth_imgs, normal_imgs, poses, ll_imgs, render_poses, [H, W, focal], i_split, i_split_not_imgs

