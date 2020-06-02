# Playground de Reconocimiento Facial

## Requirements
- Ubuntu
- Docker
- nvidia-docker2

## Docker installation
### Ubuntu:
NOTE: See: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04

```
sudo apt update
```
```
sudo apt install apt-transport-https ca-certificates curl software-properties-common
```
```
url -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
```
```
sudo apt update
```
```
apt-cache policy docker-ce
```
```
sudo apt install docker-ce
```
## Nvidia-docker2 Ubuntu Installation
Package required in the host machine to allow Docker to use NVIDIA as a runtime.
NOTE: See https://stackoverflow.com/questions/57957491/nvidia-docker-unknown-runtime-specified-nvidia

```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
## Run the program
```
docker-compose up
```
