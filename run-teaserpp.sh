#!/usr/bin/env bash
xhost +local:

set -euo pipefail

cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"


docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  teaserpp:0.0.1 \
  python ./demo_2_teaser.py

xhost -local:
