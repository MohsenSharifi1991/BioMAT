# BioMAT

<p align="center">
          <a href= "https://twitter.com/intent/tweet?text=PyTorch-VAE:%20Collection%20of%20VAE%20models%20in%20PyTorch.&url=https://github.com/AntixK/PyTorch-VAE">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" /></a>

</p>
Installation

### dataset 
Original dataset include 22 subjects while current model and result are based on 19 subjects. Subjects are from AB06 to AB25).

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.6.0 ([GitHub Repo](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32))
- CUDA enabled computing device

### Installation
```
$ git clone git@github.com:mohsensharifi1991/BioMAT.git
$ cd BioMAT
$ pip install -r requirements.txt
```
#### To Prepare Dataset from scratch
#### To use pickle dataset 
#### To use generated segmented data
#### To use existing segmented dataset
#### To run training 
#### To run hyperparameter tunning 
#### To run the Streamlit 

### References:
Transformer Resource and Lib:
1. https://timeseriesai.github.io/tsai/models.TST.html#TST
2. https://timeseriestransformer.readthedocs.io/en/latest/README.html
3. https://github.com/maxjcohen/transformer



@article{CAMARGO2021110320,
title = {A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions},
journal = {Journal of Biomechanics},
pages = {110320},
year = {2021},
issn = {0021-9290},
doi = {https://doi.org/10.1016/j.jbiomech.2021.110320},
url = {https://www.sciencedirect.com/science/article/pii/S0021929021001007},
author = {Jonathan Camargo and Aditya Ramanathan and Will Flanagan and Aaron Young},
keywords = {Locomotion biomechanics, stairs, ramps, level-ground, treadmill, wearable sensors, open dataset},
abstract = {We introduce a novel dataset containing 3-dimensional biomechanical and wearable sensor data from 22 able-bodied adults for multiple locomotion modes (level-ground/treadmill walking, stair ascent/descent, and ramp ascent/descent) and multiple terrain conditions of each mode (walking speed, stair height, and ramp inclination). In this paper, we present the data collection methods, explain the structure of the open dataset, and report the sensor data along with the kinematic and kinetic profiles of joint biomechanics as a function of the gait phase. This dataset offers a comprehensive source of locomotion information for the same set of subjects to motivate applications in locomotion recognition, developments in robotic assistive devices, and improvement of biomimetic controllers that better adapt to terrain conditions. With such a dataset, models for these applications can be either subject-dependent or subject-independent, allowing greater flexibility for researchers to advance the field.}
}