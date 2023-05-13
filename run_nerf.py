import os
import time
import cv2
import imageio
from tensorboardX import SummaryWriter
from box import Box
import yaml
from utils.utils import compute_time, make_poses_with_dummy, compute_all_metrics
from utils.configs import config_parser

# from NeRF import *
from models.nerf import *
from utils.run_nerf_helpers import *
from data_utils.data_loader import nerf_data_loader
from PIL import Image as PILImage
from tqdm import tqdm, trange

# np.random.seed(2023)
DEBUG = False
test_metric_list = ['mse', 'psnr', 'ssim', 'lpips']

def train():
    # #### Configuration ####
    parser = config_parser()
    args = parser.parse_args()
    cfg = Box.from_yaml(filename=args.config,
                        Loader=yaml.FullLoader)
    for k, v in cfg.items():
        setattr(args, k, v)
    
    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    # Load data
    K = None
    images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, extra_info = nerf_data_loader(args)

    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses)

    # Create log dir and copy the config file
    basedir = args.basedir
    tensorboardbase = args.tbdir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)

    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")
    
    args.num_images = len(images)
    # Create nerf model
    nerf = NeRFAll(args)
    
    # nerf = NeRFAll(args, kernelnet)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))

    optim_params = nerf.parameters()

    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))
    
    start = 0
    # Load Checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        smart_load_state_dict(nerf, ckpt)

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:  # args.dataset_type != 'llff' or
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
        
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start

    # Move testing data to GPU
    render_poses = torch.tensor(render_poses[:, :3, :4]).cuda()
    nerf = nerf.cuda()
    # Short circuit if only rendering out from trained model
    
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname,
                                       f"renderonly"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            poses_w_dummy, dummy_num = make_poses_with_dummy(render_poses, args.num_gpu)
            
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            nerf.eval()
            rgbshdr, disps, _ = nerf(
                hwf[0], hwf[1], K, args.chunk,
                poses=poses_w_dummy.cuda(),
                render_kwargs=render_kwargs_test,
                render_factor=args.render_factor,
            )
            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = rgbshdr
            rgbs = to8b(rgbs.cpu().numpy())
            disps = to8b(disps / disps.max())
            if args.render_test:
                for rgb_idx, rgb8 in enumerate(rgbs):
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps[rgb_idx])
                
                # evaluation
                rgbs_test = torch.tensor(rgbshdr).cuda()
                imagesf = torch.tensor(imagesf).cuda()
                rgbs_test = rgbs_test[i_test]
                target_rgb_gt = imagesf[i_test]
                
                test_vals = compute_all_metrics(rgbs_test, target_rgb_gt, test_metric_list)
                
                if isinstance(test_vals['lpips'], torch.Tensor):
                    test_vals['lpips'] = test_vals['lpips'].item()

                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"**[Evaluation]** : PSNR:{test_vals['psnr']:.8f} SSIM:{test_vals['ssim']:.8f} LPIPS:{test_vals['lpips']:.8f}\n")
                    print(f"**[Evaluation]** : PSNR:{test_vals['psnr']:.8f} SSIM:{test_vals['ssim']:.8f} LPIPS:{test_vals['lpips']:.8f}")
            else:
                prefix = 'epi_' if args.render_epi else ''
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

            return

    # ============================================
    # Prepare ray dataset if batching random rays
    # ============================================
    N_rand = args.N_rand
    train_datas = {}
    
    # if downsample, downsample the images
    if args.datadownsample > 0:
        images_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in imagesf], axis=0)
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
                        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
                        0, 0, 1]).reshape(3, 3).astype(K.dtype)
    # K = 
    # [[focal,     0, 0.5*W],
    #  [    0, focal, 0.5*H],
    #  [    0,     0,     1]]
    
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3) # [N*H*W,  ro+rd (2), 3]

    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'], train_datas['rays_y'] = xs[i_train].reshape(-1, 1), ys[i_train].reshape(-1, 1)

    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)

    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)

    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}

    print('done')
    i_batch = 0

    # Move training data to GPU
    images = torch.tensor(images).cuda()
    imagesf = torch.tensor(imagesf).cuda()

    poses = torch.tensor(poses).cuda()
    train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    time_init = time.time()
    fine_loss_weight = 0.1
    for i in trange(start, N_iters):
        # time0 = time.time()
        # Sample random ray batch
        iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()} # rays: [N_rand, ro+rd (2), 3]
        batch_rays = iter_data.pop('rays').permute(0, 2, 1) # [N_rand, 3, ro+rd (2)]

        i_batch += N_rand
        if i_batch >= len(train_datas['rays']):
            print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(len(train_datas['rays']))
            train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
            i_batch = 0

        #####  Core optimization loop  #####
        iter_data['poses'] = poses[iter_data['images_idx']].squeeze(1)
        iter_data['K'] = k_train
        nerf.train()
        rgb, _, rgb0 = nerf(H, W, K, chunk=args.chunk,
                                     rays=batch_rays, rays_info=iter_data,
                                     retraw=True,
                                     **render_kwargs_train)

        # Compute Losses
        # =====================
        target_rgb = iter_data['rgbsf'].squeeze(-2)
        img_loss = img2mse(rgb, target_rgb)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        img_loss0 = img2mse(rgb0, target_rgb)
        loss = loss + img_loss0
                
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            poses_with_dummy, dummy_num = make_poses_with_dummy(render_poses, args.num_gpu)
            
            with torch.no_grad():
                nerf.eval()
                rgbs, disps, _ = nerf(H, W, K, args.chunk, poses=poses_with_dummy.cuda(), render_kwargs=render_kwargs_test)
                
            rgbs = rgbs[:len(rgbs) - dummy_num]
            disps = disps[:len(disps) - dummy_num]
            
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs = rgbs.cpu().numpy()
            disps = (1. - disps)
            disps = disps.cpu().numpy()

            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / disps.max()), fps=30, quality=8)

            print('Done, saving', rgbs.shape, disps.shape)
            # )

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            poses_with_dummy, dummy_num = make_poses_with_dummy(poses[i_test], args.num_gpu)
            
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            
            with torch.no_grad():
                nerf.eval()
                rgbs, disps, _ = nerf(H, W, K, args.chunk, poses=poses_with_dummy.cuda(),
                               render_kwargs=render_kwargs_test)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                disps = (1. - disps)
                disps = disps[:len(disps) - dummy_num]
                disps = disps / disps.max()
                rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min()
                # saving
                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    disps8 = disps[rgb_idx].cpu().numpy()
                    disps8 = to8b(disps8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps8)

                # evaluation
                target_rgb_gt = imagesf[i_test]

                test_vals = compute_all_metrics(rgbs, target_rgb_gt, test_metric_list)
                
                if isinstance(test_vals['lpips'], torch.Tensor):
                    test_vals['lpips'] = test_vals['lpips'].item()

                tensorboard.add_scalar("Test MSE", test_vals['mse'], global_step)
                tensorboard.add_scalar("Test PSNR", test_vals['psnr'], global_step)
                tensorboard.add_scalar("Test SSIM", test_vals['ssim'], global_step)
                tensorboard.add_scalar("Test LPIPS", test_vals['lpips'], global_step)
                
            with open(test_metric_file, 'a') as outfile:
                outfile.write(f"iter{i}/globalstep{global_step}: MSE:{test_vals['mse']:.8f} PSNR:{test_vals['psnr']:.8f}"
                              f" SSIM:{test_vals['ssim']:.8f} LPIPS:{test_vals['lpips']:.8f}\n")
                print(f"**[Evaluation]** Iter{i}/globalstep{global_step}: MSE:{test_vals['mse']:.8f} PSNR:{test_vals['psnr']:.8f}"
                              f" SSIM:{test_vals['ssim']:.8f} LPIPS:{test_vals['lpips']:.8f}")
            
            print('Saved test set')

        if i % args.i_tensorboard == 0:
            tensorboard.add_scalar("Loss", loss.item(), global_step)
            tensorboard.add_scalar("PSNR", psnr.item(), global_step)
            # for k, v in extra_loss.items():
                # tensorboard.add_scalar(k, v.item(), global_step)

        if i % args.i_print == 0:
            dt_h, dt_m, dt_s = compute_time((time.time() - time_init))
            dt_h, dt_m = int(dt_h), int(dt_m)
            # print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            # print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} TIME: {dt_h}h:{dt_m}m:{dt_s:.2f}s")
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} TIME: {dt_h}h:{dt_m}m:{dt_s:.2f}s")
            
        global_step += 1
        
    with open(test_metric_file, 'a') as outfile:
        outfile.write(f"TRINING TIME: {dt_h}h:{dt_m}m:{dt_s:.2f}s")     

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

