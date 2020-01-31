# Faster-RCNN implemented in pytorch
Faster-RCNN implementation in pytorch. It is described in this article: https://arxiv.org/abs/1506.01497 

## Installation
In order to run this repo, you need to have docker installed. You can find the installation instruction here: https://www.docker.com/#/get_started .
After docker is installed I **strongly recommend** you to install nvidia-docker and use it with cuda. Otherwise the training or the evaluation will be too slow to use. You can find the installation instructions at the nvidia-docker website: https://github.com/NVIDIA/nvidia-docker. With that in mind, you either keep this version of pytorch or change to the nvidia-docker enabled version of the base image and change the dockerfile. 

## Datasets
For training and evaluation purposes, you need to have some datasets prepared.
