# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.12-py3
#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.08-py3
FROM ${FROM_IMAGE_NAME}

ENV PYTHONPATH /workspace/hifigan
WORKDIR /workspace/hifigan

#--- add my user so new files/folders won't be created with root as owner (itamark)
RUN groupadd --gid 1008 itamark && adduser --gid 1008 --uid 1008 --disabled-password --gecos '' itamark

ADD requirements.txt .
RUN pip install nvidia-pyindex
RUN pip install -r requirements.txt

COPY . .
