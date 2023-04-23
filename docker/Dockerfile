FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /root

RUN apt update \
    && apt upgrade -y \
    && apt install -y git \
    && apt install -y vim \
    && apt install -y libgl1 libglib2.0-0

RUN apt clean \
    && rm -rf /var/lib/apt/lists/* 