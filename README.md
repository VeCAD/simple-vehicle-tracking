DOCUMENTATION
=============
# 1. Contents:
```
.
├── /yolo_weights_params
├── /vehicle_track_source
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
```

# 2. Setup Guide

Install [Docker](https://docs.docker.com/engine/install/ubuntu/),[NVIDIA docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide) and [docker-compose](https://docs.docker.com/compose/install/) on a host Ubuntu system with NVIDIA GPU and respective drivers installed.

Due to file size limit, download the Yolo3 pretrained weights here https://pjreddie.com/media/files/yolov3.weights \
and place it in /yolo_weights_params folder.

## How to build and run container
```sh
# Build the image
cd ~/simple-vehicle-tracking && docker-compose build

# Run the container
docker-compose run vehicle bash

#Run the program
python3 vehicle_track.py
```

Run the following if CV window failes to display on host PC
```sh
No protocol specified
Unable to init server: Could not connect: Connection refused

(UI:142): Gtk-WARNING **: 10:03:12.231: cannot open display: :1

xhost +local:docker
```


# 3. Comments

3.1 Existing available libraries/algortihms used, credits where its due, GPL inherited from the libraries/repos:\
  a.Using Darknet Yolo AI model by https://pjreddie.com/darknet/ , https://github.com/pjreddie/darknet with yolov3.weights pretrained weights and network configuration in yolov3.cfg to perform vehicle detection and recognition on the input frame.  \
  b.The sort tracker is based on SORT: A Simple, Online and Realtime Tracker from https://github.com/abewley/sort with default parameters.\
3.2. Track accuracy could be improved using more robust tracker Eg. DeepSor t\
3.3. Performance improvement by threading processes
