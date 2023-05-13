import numpy as np
from data_utils.load_llff import load_llff_data
from data_utils.load_blender import load_blender_data, load_blender_time_data


def nerf_data_loader(args):
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args, args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify,
                                                                  path_epi=args.render_epi)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        print('LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.min(bds) * 0.9
            far = np.max(bds) * 1.0

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
        
        return images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, None
        
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
    
        return images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, None
    
    elif args.dataset_type == 'blender_time':
        images, poses, render_poses, hwf, i_split, times = load_blender_time_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        time_train, time_val, time_test = times[i_split[0]], times[i_split[1]], times[i_split[2]]
        
        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
        
        time_num = len(i_train)

        extra_info = {'time_train': time_train, 'time_val': time_val, 'time_test': time_test, 'time_num': time_num}
        
        return images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, extra_info
    
    else:
        raise Exception('Unknown dataset type: ', args.dataset_type)