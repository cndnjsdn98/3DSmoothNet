#!/usr/bin/env bash
xhost +local:

set -euo pipefail

cd -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# docker run --rm -it \
#   --gpus all \
#   --shm-size=16g \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#   -v "$(pwd)":/workspace/3dsmoothnet \
#   -w /workspace/3dsmoothnet \
#   teaserpp:0.0.1 \

python ./subsample_pcl.py @./scripts/downsample_config.txt @./scripts/keypoints_config.txt @./scripts/battery_pack_meta.txt

python ./compute_keypoints.py @./scripts/keypoints_config.txt @./scripts/battery_pack_meta.txt

docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  --pid=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/3dsmoothnet \
  -w /workspace/3dsmoothnet \
  3dsmoothnet:0.0.1 \
  python ./compute_descriptors.py  @./scripts/descriptor_config.txt @./scripts/input_parametrization_config.txt @./scripts/battery_pack_meta.txt

python ./perform_teaserpp.py @./scripts/input_parametrization_config.txt @./scripts/teaserpp_config.txt @./scripts/battery_pack_meta.txt

xhost -local:
