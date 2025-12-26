# AUTOENCODER-CUDA-

## Installl CUDA TOOLKIT 
https://developer.nvidia.com/cuda-downloads

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1

## Install CuDNN 
https://developer.nvidia.com/cudnn-downloads

wget https://developer.download.nvidia.com/compute/cudnn/9.17.0/local_installers/cudnn-local-repo-ubuntu2404-9.17.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.17.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.17.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn9-cuda-12


## Data

## Run file
nvcc ./main.cu  ./gpu_autoencoder.cu ./utils.cpp  -o main


build file:
'''bash
rm -r build
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
cmake --build . -j8
cd ..
./build/gpu_main --epoch=1
'''
