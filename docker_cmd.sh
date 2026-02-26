xhost +local:

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --pid=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  wonoo/3dsmoothnet_teaser:dev \
  bash

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --pid=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  wonoo/3dsmoothnet_teaser:dev \
  python ./demo.py

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --pid=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  wonoo/3dsmoothnet_teaser:dev \
  python ./demo_2.py


docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --pid=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  wonoo/3dsmoothnet_teaser:dev \
  /opt/conda310/bin/conda run -n py310 python ./demo_2_teaser.py