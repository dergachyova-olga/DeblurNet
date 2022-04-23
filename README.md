# Deep learning for sodium MRI deblurring and denoising (2020)#


This is a tool for processing sodium MRI using deep learning. In its current stage the code implements deblurring and denoising functionalities using convolutional neural networks (CNN). However, the code can serve as base for implementing other types of MRI processing. 
The code includes a full processing pipeline: data loading and preparation, CNN architecture creation and training from scratch, model validation, storage, and inference.


### System requirements ###

The code is written in Python 3 and uses TensorFlow framework for CNN implementation. It can be run in two modes: 1) CPU only (no acceleration), and 2) GPU (with acceleration).
The CPU only version requires installing tensorflow package (cpu version) and some additional dependencies (please refer to *requirements.txt* file). The GPU version requires another version of tensorflow package with gpu support.
In addition, compatible Nvidia drivers and Cuda libraries have to be installed. 

Some useful links:

* [Tensorflow installation](https://www.tensorflow.org/install/pip)
* [Cuda installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [Nvidia drivers](https://www.nvidia.com/Download/index.aspx)

*Warning: Tensorflow is a rapidly evolving framework and this code may not run with its newer versions. Often, there can also be major issues with inter-compatibility of gpu-related dependencies.*

The code was developed and tested using the following configuration:

* Python 3.6.8

* TensorFlow (gpu) 1.14.0

* Cuda 10.1.243

* Cudnn 7.6.5

* Nvidia driver 430.64

* Nvidia GeForce RTX 2080 Ti GPU


### How to run the code? ###

To run the code execute main.py script using Python 3 interpreter from the folder in which it is situated. 

ex: `python3 main.py` 

Prior main script execution populate *configs/config.ini* file with desired parameters. A sample of a configuration file is provided in *configs/config_sample.ini*. All the fields have to be set.
This configuration file will be automatically loaded as default as the first command in the main script. Another option is to directly provide a desired configuration file as script's input argument. 

ex: `python3 main.py configs/config.ini`


### Code structure ###

At the upper level, besides main.py script the code also contains config.py script that loads the provided configuration and creates an object containing all the parameters used by the main.py script.

The code also includes 5 sub-directories:

*   **configs** - stores configuration files,

*   **convnet** - contains core scripts and classes defining how data is loaded and network is trained, as well as general network building blocks (layers and cost functions),

*   **networks** - contains architectures of implemented networks,

*   **scripts** - bash scripts that can be used outside main.py to prepare data or automatically run experiments with different configurations,

*   **utils** - utilities used in main.py script to mostly format and store data.


### Available networks ###

There are 4 available networks:

*   **Unet** (*networks/unet.py*) - classic [Unet network](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) modified to map input image(s) to a desired processed (denoised or deblurred) output image.

*   **C2net** (*networks/c2net.py*) - two-channel version of the present Unet, where two input modalities are processed separately and only concatenated at the end to be mapped to output.

*   **Offresnet** (*networks/offresnet.py*) - implementation of [Offresnet](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27825), based on a classic [resnet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) architecture, and adapted for deblurring and denoising. 

*   **DnCNN** (*networks/DnCNN.py*) - implementation of [denoising convolutional neural network (DnCNN)](https://ieeexplore.ieee.org/abstract/document/7839189).


### Data ###

The code primarily works with imaging data provided in a form of (MATLAB) matrices (i.e., *.mat* files). However, a new class inheriting from `BaseDataProvider` in *convnet/loader.py* can be easily added to support other data formats.
Output data defines the task that is desired to be performed (deblurring, denoising or other). The implemented networks only map provided input to provided output.


### Author ###

Olga Dergachyova (olga.dergachyova@nyulangone.org). Last update: Oct 2020.

