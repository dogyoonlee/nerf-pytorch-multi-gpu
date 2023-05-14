# Pytorch Implementation of NeRF for Multi-GPU without DDP.

Modified nerf code based on pytorch to use the multi gpu without complex DDP process.

We simply use `nn.DataParallel` based on the [[DeblurNeRF]](https://github.com/limacv/Deblur-NeRF) and [[DP-NeRF]](https://github.com/dogyoonlee/DP-NeRF) code referring the [[nerf-pytorch]](https://github.com/yenchenlin/nerf-pytorch) code.

# Training & Evaluation

## 1. Environment
```
git clone https://github.com/dogyoonlee/nerf-pytorch-multi-gpu.git
cd nerf-pytorch-multi-gpu
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>
  <li>numpy
  <li>scikit-image
  <li>torch>=1.11
  <li>torchvision>=0.12.0
  <li>imageio
  <li>imageio-ffmpeg
  <li>matplotlib
  <li>configargparse
  <li>tensorboardX>=2.0
  <li>opencv-python
  <li>einops
  <li>tensorboard
  <li>python-box
  <li>pyyaml
  <li>tqdm
  <li>mediapy
  <li>pillow
</details>

## 2. Download dataset
There are various datasets based on NeRF

- NeRF dataset: [[Github]](https://github.com/bmild/nerf) [[Dataset]](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

- HDR-NeRF dataset: [[Github]](https://github.com/xhuangcv/hdr-nerf) [[Dataset]](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I)

- Deblur-NeRF dataset: [[Github]](https://github.com/limacv/Deblur-NeRF) [[Dataset]](https://hkustconnect-my.sharepoint.com/personal/lmaag_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flmaag%5Fconnect%5Fust%5Fhk%2FDocuments%2Fshare%2FCVPR2022%2Fdeblurnerf%5Fdataset&ga=1)

We provide config example in `configs` folder.

Create your own config file referring the provided file.


## 3. Set parameters
Change the training parameters in config file to train on your computer.

You must set the `config` and `expname` parameter to run the training.

## 4. Train
For example, to train `lego` scene in `nerf_synthetic`,

```
python run_nerf.py --config ./configs/nerf_synthetic/tx_nerf_synthetic_lego.txt --expname <experiment_name>
```

The training and tensorboard results will be save in `<basedir>/<expname>` and `<tbdir>`.

## 5. Evaluation

Evaluation is automatically executed every `--i_testset` iterations.
Please refer the other logging options in `configs.py` to adjust save and print the results.

After the training, execute the evaluation results following command.
For example, to evaluate `lego` scene after 200000 iteration,

```
python run_nerf.py --config ./configs/nerf_synthetic/tx_nerf_synthetic_lego.txt --expname <dir_to_log> --ft_path ./<basedir>/<expname_trained>/200000.tar --render_only --render_test
```


# Visualization of trained model
You can render or save the results after 200000 iteration of training following process.

## 1. Visualize the trained model as videos of spiral-path.
Results will be saved in `./<basedir>/<dir_to_log>/renderonly_path_199999`.

```
python run_nerf.py --config ./configs/nerf_synthetic/tx_nerf_synthetic_lego.txt --expname <dir_to_log> --ft_path ./<basedir>/<expname_trained>/200000.tar --render_only 
```

## 2. Visualize the trained model as images of training views in datasets.
Results will be saved in `./<basedir>/<dir_to_log>/renderonly_test_199999`

```
python run_dpnerf.py --config ./configs/nerf_synthetic/tx_nerf_synthetic_lego.txt --expname <dir_to_log> --ft_path ./<basedir>/<expname_trained>/200000.tar --render_only  --render_test
```

# Notes

## 1. The number of GPU

If you want to set the number of GPU, change `num_gpu` value.

## 2. GPU memory 

If you have not enough memory to train on your single GPU, set `N_rand` to a smaller value, or use multiple GPUs.



# License

MIT License


# References/Contributions

- [nerf-official Code](https://github.com/bmild/nerf)

- [nerf-pytorch Code](https://github.com/yenchenlin/nerf-pytorch)

- [Deblur-NeRF Code](https://github.com/limacv/Deblur-NeRF)

- [DP-NeRF Code](https://github.com/dogyoonlee/DP-NeRF)
