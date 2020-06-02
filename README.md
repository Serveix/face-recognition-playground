# Playground de Reconocimiento Facial

## Requirements
- Ubuntu
- Docker

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
### Docker group permissions
```
sudo groupadd docker
```
```
sudo usermod -aG docker $USER
```
Then logout and log back in!

## Run the program
```
docker-compose up
```
