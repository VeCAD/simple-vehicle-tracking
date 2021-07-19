FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    build-essential \
    git\
    unzip \
    pkg-config \
    python-setuptools \
    python-dev \
    libomp-dev \
    python3-opencv \
    nano \
    wget \
    python3-pip \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    qt5-default \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

RUN pip3 install --upgrade setuptools pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /vehicle

RUN git clone https://github.com/pjreddie/darknet.git

WORKDIR darknet/

RUN mkdir -p output

RUN sed -i 's/GPU=.*/GPU=1/' Makefile && \
    sed -i 's/CUDNN=.*/CUDNN=1/' Makefile && \
    sed -i 's/OPENMP=.*/OPENMP=1/' Makefile && \
    make

WORKDIR /vehicle/track
