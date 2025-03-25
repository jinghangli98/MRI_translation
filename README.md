# T1w to TSE MRI Image Translation using Diffusion Models

This repository contains the implementation of a diffusion-based model for translating T1-weighted MRI images to TSE (Turbo Spin Echo) MRI images. The model uses a conditional diffusion process with U-Net architecture.

## Installation

```bash
# Clone the repository
git clone https://github.com/jinghangli98/MRI_translation.git
cd MRI_translation

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The model is designed to work with paired T1w and TSE MRI images. The dataset should be organized in the following structure:

```
data_path/
├── train/
│   ├── t1w/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── tse/
│       ├── img1.png
│       ├── img2.png
│       └── ...
└── val/
    ├── t1w/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── tse/
        ├── img1.png
        ├── img2.png
        └── ...
```

## Usage

### Single GPU Training

```bash
python train.py --batch_size 32 --resize_size 64 --crop_size 448 --max_epochs 1000 --log True
```

### Distributed Training

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py --distributed --batch_size 8 --log True

# Manual setup
python train.py --distributed --world_size 4 --batch_size 8 --log True
```

### Key Arguments

- `--lr`: Learning rate (default: 1e-5)
- `--crop_size`: Size to crop images to (default: 448)
- `--resize_size`: Size to resize images to (default: 64)
- `--data_path`: Path to the dataset directory
- `--batch_size`: Batch size for training (default: 32)
- `--max_epochs`: Maximum number of training epochs (default: 1000)
- `--sample`: Number of samples to use from dataset (default: 100)
- `--val_interval`: Interval for validation (default: 1)
- `--log`: Enable wandb logging metrics (default: False)
- `--use_deepcache`: Use Deep Cache for accelerated inference (default: True)
- `--eval_num`: Number of images to evaluate (default: 1)
- `--checkpoint_path`: Path to save or load model checkpoint

### Distributed Training Arguments

- `--distributed`: Enable distributed training with DDP
- `--local_rank`: Local rank for distributed training
- `--world_size`: Number of processes/GPUs for distributed training
- `--dist_url`: URL used to set up distributed training
- `--dist_backend`: Distributed backend to use (nccl, gloo, etc.)
- `--master_addr`: Master address for distributed training
- `--master_port`: Master port for distributed training

## Model Architecture

The model is based on MONAI's implementation of the diffusion U-Net with the following configuration:

- 2D spatial dimensions
- 2 input channels (T1w + noise)
- 1 output channel (TSE)
- Channel sizes: (256, 256, 512)
- Attention at the deepest level
- 2 residual blocks per level
- 512 attention head channels

## Training Process

1. The model takes a concatenated input of a T1w MRI image and noise
2. It predicts the noise added to the target TSE image
3. During inference, a random noise image is gradually denoised conditioned on the input T1w image

## Evaluation Metrics

The model is evaluated using:
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Learned Perceptual Image Patch Similarity (LPIPS)

## Result Visualization

Generated images are saved in the `visualization_results` directory. Each saved image contains:
- Input T1w image
- Generated TSE image
- Ground truth TSE image

## Checkpoints

Checkpoints are saved in the `checkpoints/{resize_size}/` directory. The model with the best SSIM score is saved as `t1w_to_tse_model_{resize_size}_ssim_{best_ssim}.pt`.

## License


## Citation

If you use this code in your research, please cite:


## Acknowledgements

- [MONAI](https://github.com/Project-MONAI/MONAI) for providing the diffusion model implementation
- [Diffusion Models for Medical Image Generation](https://github.com/Project-MONAI/tutorials/blob/main/modules/generative/diffusion_model.ipynb) for the initial inspiration
