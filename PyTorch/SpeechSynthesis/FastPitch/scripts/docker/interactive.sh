#!/usr/bin/env bash

PORT=${PORT:-8888}
USER=${DOCKER_USER:=root}

docker run --user $USER:$USER --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -v $PWD:/workspace/fastpitch/ fastpitch:latest bash 
