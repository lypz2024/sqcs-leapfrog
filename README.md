# Leapfrog Solver and SqueezeNet Cosine Similarity (SqCS) for Faster and Quality Sampling

## Leapfrog Solver

- We develop the Leapfrog Solver for faster sampling of diffusion models, achieving high-quality results in approximately 4 inference steps. 

<div align="center">
    <img src="Imagenet_samples.png" alt="Imagenet samples">
    <p>The figure above shows the images sampled on ImageNet dataset.</p>
</div>

<div align="center">
    <img src="unconditional_samples.png" alt="Unconditional samples">
    <p>The figure above shows the unconditional images sampled on LSUN Bedrooms, LSUN Churches, CelebAHQ, and FFHQ datasets.</p>
</div>

<div align="center">
    <img src="samples_all_solvers.png" alt="Samples of all solvers">
    <p>The figure above shows the comparison of sampled images by various solvers in 4 inference steps.</p>
</div>

- The Leapfrog Solver works as depicted below:

<div align="center">
    <img src="leapfrog_solver.png" alt="Leapfrog Solver">
</div>

### Features

- **Efficient Sampling**: Accelerates sampling
- **Versatile Application**: Generates high-quality samples for both class-conditional and unconditional (without class) images.

### Usage
- We implemented our Leapfrog Solver in the code bleow:
```shell script
leapfrog.py
```  
- Set up a conda environment and install below libraries:

```shell script
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
```

- To generate ImageNet samples (conditional images), run:

```shell script
python imagenet_sampling.py
```

- To generate unconditional images for datasets like CelebAHQ, FFHQ, LSUN Churches, and LSUN Bedrooms, use:

```shell script
python sample_diffusion.py -r <path for model.ckpt> -l <output directory for sampled images> -n <number of samples to be generated> --batch_size <batch size> -c <number of inference steps> -e <eta>
```

- Example to generate samples of CelebAHQ dataset:
```shell script
python sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l output_samples/4steps -n 50000 --batch_size 100 -c 4 -e 0
```
- We utilized the pre-trained models weights of LDMs (Latent diffusion models) from  [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).

## SqueezeNet Cosine Similarity (SqCS)

- We develop SqCS to assess the quality of generated images across several inference steps and to determine the best inference step. It address the several drawbacks of existing popular metrics like FID,SSIM, and PSNR.

- We compute SqCS metric as shown in this flowchart.

<div align="center">
    <img src="flowchart_sqcs.png" alt="SqCS">
</div>

### Usage

- To compute SqCS metric between real and generated images, run:

```shell script
python sqcs.py <directory of real images> <directory of generated images> --batch_size <batch size>
```

- Example:
```shell script
python sqcs.py data/real_images/lsun_churches output_samples/lsun_churches --batch_size 500
```



