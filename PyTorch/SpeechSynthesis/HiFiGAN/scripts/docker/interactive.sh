#!/usr/bin/env bash

#--- in order to run the docker using my user:
#   >> DOCKER_USER=itamark bash scripts/docker/interactive.sh

PORT=${PORT:=8888}
USER=${DOCKER_USER:=root}

#docker run --gpus=all -it --rm --user $(id -u):$(id -g) -e CUDA_VISIBLE_DEVICES --ipc=host -v $PWD:/workspace/hifigan/ hifigan:latest bash 
#docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -v $PWD:/workspace/hifigan/ hifigan:latest bash 
docker run --user $USER:$USER --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -v $PWD:/workspace/hifigan/ hifigan:latest bash 
