
## CryoGEM💎 : Physics-Informed Generative Cryo-Electron Microscopy

Anonymous authors 

<p align="center">
  <img src="assets/teaser.jpg", width=650>
</p>

## 🔧 Dependencies and Installation
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    cd cryoGEM
    ```

1. Install cryoGEM

    ```bash 
    pip install -e . # linux only
    ```

---

# Prepare Clean Micrographs
```console
Usage: genem gen_data [options] ...
  -h                              show help
  --device                        device to run the code [cuda:0]
  --save_dir                      directory to save the projection micrographs
  --mode                          mode of the dataset [homo | hetero]
  --input_map                     (homo) input map file 
  --symmetry                      (homo) symmetry of volume [C1, D7...]
  --drgn_dir                      (hetero) cryodrgn result directory
  --drgn_epoch                    (hetero) cryodrgn checkpoint index
  --n_micrographs                 number of micrographs to generate [10]
  --micrograph_size               micrograph size [1024]
  --particles_mu                  (Gaussian sampler) mean particles per micrograph
  --particles_sigma               (Gaussian sampler) sigma of distribution
  --particle_collapse_ratio       larger -> more dense particles (0~1)
  --mask_threshold                threshold for particle mask   
Example(homo):
  # training dataset
  genem gen_data --mode homo --device cuda:0 \
  --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc \
  --save_dir save_images/gen_data/Ribosome\(10028\)/training_dataset/ \
  --n_micrographs 100 --particle_size 90 --mask_threshold 0.9
  # testing dataset
  genem gen_data --mode homo --device cuda:0 \
  --input_map testing/data/exp_abinitio_volumes/densitymap.10028.90.mrc  \
  --save_dir save_images/gen_data/Ribosome\(10028\)/testing_dataset/ \
  --n_micrographs 1000 --particle_size 90 --mask_threshold 0.9
Example(hetero):
  # training dataset
  genem gen_data --mode hetero --device cuda:0 --drgn_epoch 49 \
  --drgn_dir testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result \
  --save_dir save_images/gen_data/Integrin\(10345\)/training_dataset/ \
  --n_micrographs 100 --particle_size 100 --mask_threshold 0.7 
  # testing dataset
  genem gen_data --mode hetero --device cuda:0 --drgn_epoch 49 \
  --drgn_dir testing/data/exp_abinitio_volumes/10345_neural_volume/drgn_result \
  --save_dir save_images/gen_data/Integrin\(10345\)/testing_dataset/ \
  --n_micrographs 1000 --particle_size 100 --mask_threshold 0.7 
```
<div class="center">

| Dataset | Symmetry | Micrograph $n_{train}/n_{test}$ | Micrograph $\text{N}\times\text{N}$ | Particles $\mu$ | Particles $\sigma$ | Particle collapse ratio | Mask threshold |
|:-----------------:|:--:|:---------:|:---------:|:------:|:-----:|:----:|:---:|
| Proteasome(10025) | D7 | 100/1000  | 1024x1024 | 245.88 | 33.62 | 0.75 | 0.9 |
| Ribosome(10028)   | C1 | 100/1000  | 1024x1024 |  97.52 | 13.64 | 0.50 | 0.9 |
| Integrin(10345)   | C1 | 100/1000  | 1024x1024 |  42.41 | 17.45 | 0.40 | 0.7 |
| PhageMS2(10075)   | C1 | 100/1000  | 1024x1024 |  35.42 | 17.74 | 0.50 | 0.9 |
| HumanBAF(10590)   | C1 | 100/1000  | 1024x1024 | 150.44 | 17.06 | 0.55 | 0.9 |

</div>

# Estimate Ice Gradient (Real)
```console
Usage: genem esti_ice [options] ... 
  -h                              show help
  --apix                          pixel size (Angstrom) [1.0]
  --input_dir                     (required) input micrograph directory
  --save_dir                      (required) save directory
  --output_len                    length if estimated ice gradient [1024]
  --device                        device to run the code [cuda:0]
Example:
  genem esti_ice --apix 5.36 --device cuda:0 \
  --input_dir testing/data/Ribosome\(10028\)/reals/ \
  --save_dir save_images/esti_ice/Ribosome\(10028\)/ 
```
<div class="center">

| Dataset | Apix | Ice Gradient $\text{N}\times\text{N}$ |
|:-----------------:|:----:|:---------:|
| Proteasome(10025) | 4.62 | 1024x1024 | 
| Ribosome(10028)   | 5.36 | 1024x1024 |
| Integrin(10345)   | 4.04 | 1024x1024 |
| PhageMS2(10075)   | 4.64 | 1024x1024 |
| HumanBAF(10590)   | 4.50 | 1024x1024 |

</div>

# Train CryoGEM Model
If you want to use your own dataset or model, please modify the dataset and model configuration in the [config file](genem/config.py) and create custom dataset according to [template dataset](genem/datasets/template_dataset.py) and custom model according to [template model](genem/models/template_model.py). 
```console
Usage: genem train [options] ...
  -h                              show help
  --name                          experiment name [empiar-10028-test]
  --gpu_ids                       GPU ids, -1 for CPU [0]
  --ngf                           # of gen filters in first conv [64]
  --ndf                           # of discrim filters in first conv [128]
  --netD                          discriminator [basic | n_layers | pixel]
  --n_layers_D                    only used if netD==n_layers [3]
  --netG                          generator architecture [unet_1024]
  --norm                          instance/batch normalization [instance]
  --no_dropout                    no dropout for the generator [True]
  --batch_size                    input batch size [1]
  --crop_size                     cropped image size [1024]
  --n_epochs                      epochs with initial learning rate [25]
  --n_epochs_decay                epochs to linearly to zero [75]
  --lr                            initial learning rate for adam [1e-4]
  --lr_policy                     learning rate policy.[linear]
  --max_dataset_size              maximum loading images [100]
  --apix                          pixel size (Angstrom) [5.36]
  --real_dir                      directory of real images 
  --sync_dir                      directory of sync images 
  --mask_dir                      directory of particle masks
  --weight_map_dir                directory of real ice gradients
Example:
  genem train --name empair-10028-test --max_dataset_size 100 --apix 5.36 --gpu_ids 0 \
  --real_dir testing/data/Ribosome\(10028\)/reals/ \
  --sync_dir save_images/gen_data/Ribosome\(10028\)/training_dataset/mics_mrc \
  --mask_dir save_images/gen_data/Ribosome\(10028\)/training_dataset/particles_mask \
  --weight_map_dir save_images/esti_ice/Ribosome\(10028\)/ 
```

# Test CryoGEM Model
```console
Usage: genem test [options] ...
  -h                              show help
  --name                          experiment name [empiar-10028-test]
  --gpu_ids                       GPU ids, -1 for CPU [0]
  --ngf                           # of gen filters in first conv [64]
  --netG                          generator architecture [unet_1024]
  --norm                          instance/batch normalization [instance]
  --batch_size                    input batch size [1]
  --crop_size                     cropped image size [1024]
  --num_test                      number of generated images [1000]
  --max_dataset_size              maximum loading images [1000]
  --apix                          pixel size (Angstrom) [5.36]
  --sync_dir                      directory of sync images
  --mask_dir                      directory of particle masks
  --pose_dir                      directory of particle location 
  --weight_map_dir                directory of real ice gradients
  --save_dir                      directory to save the generated images
  --generate_shift                if true, store shifted particles [False]
  --pixel_shift_max               max pixel for particle shift [5]
Example:
  genem test --name empair-10028-test \
  --max_dataset_size 1000 --num_test 1000--apix 5.36 --gpu_ids 0 \
  --sync_dir save_images/gen_data/Ribosome\(10028\)/testing_dataset/mics_mrc \
  --mask_dir save_images/gen_data/Ribosome\(10028\)/testing_dataset/particles_mask \
  --pose_dir save_images/gen_data/Ribosome\(10028\)/testing_dataset/mics_particle_info \
  --weight_map_dir save_images/esti_ice/Ribosome\(10028\)/ \
  --save_dir save_images/test/Ribosome\(10028\)/ 
```

# Tutorial

The tutorials are presented in Jupyter notebooks. Please install Jupyter following the instructions [here](http://jupyter.org/install).

1. [EMPIAR 10028 guide](testing/empiar-10028.ipynb)

To run the tutorial steps on your own system, you will need to install [Jupyter](http://jupyter.org/install).

With Anaconda this can be done with:
```
conda install jupyter
```