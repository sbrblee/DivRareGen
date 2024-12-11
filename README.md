# Diverse Rare Sample Generation with Pretrained GANs
Repository for Diverse Rare Sample Generation with Pretrained GANs (***AAAI 2025***)

<img width="1675" alt="image" src="https://github.com/user-attachments/assets/8fa76f32-f63b-492d-96dd-19f82ee7a73b">
<img width="1657" alt="image" src="https://github.com/user-attachments/assets/e218c861-cd37-4322-954e-44d84511c1f9">

## Installation
The environment was builded on [StyleGAN2-pytorch by rosinality](https://github.com/rosinality/stylegan2-pytorch).

With Anaconda, environment can be created as follows:
```
conda env create --file environment.yml 
```

## Getting Started
### Feature Extraction
```
python -u scripts/feature_extraction.py --data_path {image directory} # The default model option is VGG16.
```
This code generates the feature vectors from a given image directory in npz file format.

### Train Density Estimator: Normalizing Flow Training
```
python -u scripts/nf_train.py --npz_path {path to real feature npz file}
```
This code trains the normalizing flow model (Glow model) from the features in a given npz file. We provide the [Glow model](https://proceedings.neurips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html) architecture used in our paper in `models` folder as default. This code also saves the MinMax scaler for the feature vectors.

### Diverse Rare Sample Generation
```
python -u scripts/divrare_optimization.py --zG_path {path to reference latent vectors npy file} --real_feature_path {path to real feature npz file} --nf_ckpt {path to checkpoint of normalizing flow model} --scaler_path {path to scaler} --dists_path {path to penalizing distances}
```
This code generates diverse rare samples for given reference latent vectors. The following options will be helpful to control the algorithm hyperparameters:

`--n_sample` number of rare samples to generate per reference

`--rand_scale` scale of noise to add to the initial latent vector for multi-start approach

`--lambda1` coefficient of the similarity objective

`--lambda2` coefficient of the diversity objective


