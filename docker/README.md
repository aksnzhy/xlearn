## Introduction
A tiny Dockerfile for building an image for `xlearn` in Python 3.6. 

## Prerequisites
* Docker 

## Build a Docker image
After cloning the `xlearn` repository from Github, run the following,
```shell
cd xlearn/docker
docker build -t <your_image_name> .
```

## Run the built Docker image in a container
```shell
docker run -it <your_image_name>
```
In the shell, go to the `xlearn` and run the demo scripts in there. 
