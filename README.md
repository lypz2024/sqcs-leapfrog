# SqueezeNet Cosine Similarity for Faster Sampling with Leapfrog Solver

# Leapfrog Solver
We develop Leapfrog Solver for faster sampling of diffusion models in ~ 4 inference steps.
![Sampled Images](sampled_images.png)
Our leapfrog solver works as shown in this figure ![Leapfrog solver](leapfrog_solver.png)
We apply leapfrog solver to Stable Diffusion models to accelarate sampling and generate high quality samples for both class conditional and unconditional (without class) images. If you want to generate ImageNet samples (conditional images), run this command
python imagenet_sampling.py
If you want to generate unconditional images of datasets like - CelebAHQ, FFHQ, LSUN churches, and LSUN Bedrooms, run this command
python sample_diffusion.py -r <path for model.ckpt> -l <output directory for sampled images> -n <number of samples> --batch_size <batch size> -c <number of inference steps> -e <eta>
An example is shown below
python sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l output_samples/4steps -n 50000 --batch_size 100 -c 4 -e 0

# SqueezeNet Cosine Similarity (SqCS)
We develop SqCS to assess the quality of generated images across several inference steps and to determine the best inference step. It address the several drawbacks of existing popular metrics like FID,SSIM, and PSNR.

