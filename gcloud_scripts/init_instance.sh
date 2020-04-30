#!/bin/bash

#machine: n1-standard-2 :2 vCPU, 7,5 Go memory
#image: c0-common-gce-gpu-image-20200128
#GPU: 1 x NVIDIA Tesla K80

wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh && bash Anaconda3-5.2.0-Linux-x86_64.sh -bfp /usr/local
pip install --upgrade pip
pip install -r requirements-dev.txt
sudo timedatectl set-timezone Europe/Paris

#Enable bash script to run sudo commands without password
sudo visudo
#Add 'username ALL=(ALL) NOPASSWD:ALL', save and close the file