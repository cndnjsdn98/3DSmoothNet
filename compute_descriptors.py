# demo.py ---
#
# Filename: demo.py
# Description: Demo of the 3DSmoothNet pipeline. 
# Comment: Some functions adapated from the open3d library http://www.open3d.org/
#
# Author: Gojcic Zan, Zhou Caifa
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Paper: https://arxiv.org/abs/1811.06879
# Created: 03.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:


import subprocess
import open3d as o3d
import glob
import json
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

from core import config
from core import network

try:
    reg = o3d.pipelines.registration
except AttributeError:
    reg = o3d.registration

FRAG1_COLOR = [1.0, 0.75, 0.0]
FRAG2_COLOR = [0, 0.629, 0.9]
SPHERE_COLOR_1 = [0,1,0.1]
SPHERE_COLOR_2 = [1, 0.3, 0.05]
NOISE_BOUND = 0.01
N_OUTLIERS = 20
OUTLIER_TRANSLATION_LB = 0.001
OUTLIER_TRANSLATION_UB = 0.05


def main(config_arguments):
    print("\n-------------------------")
    # Run the input parametrization
    config_arguments.input_pcl_folder = os.path.join(config_arguments.input_pcl_folder, "subsampled")
    point_cloud_files = glob.glob(os.path.join(config_arguments.input_pcl_folder, '*.ply'))
    if len(point_cloud_files) != 2:
        print("Check Number of files in {}. There should be 2 .ply files".format(config_arguments.input_pcl_folder))
        return
    keypoints_files = [point_cloud_files[0]+"_keypoints", point_cloud_files[1]+"_keypoints"]

    # Load reference point cloud
    voxel_grid = config_arguments.voxel_grid # Half size of the voxel grid in the unit of the point cloud. Defaults to 0.15.
    n_voxels = config_arguments.n_voxels # Number of voxels in a side of the grid. Whole grid is nxnxn. Defaults to 16.
    gaussian_width = config_arguments.gaussian_width # Width of the Gaussia kernel used for smoothing. Defaults to 1.75.
    config_arguments.input_dim = n_voxels**3
    # Clear evaluate folders
    sdv_dir = os.path.join(config_arguments.input_pcl_folder, "sdv", "*.csv")
    output_dir = os.path.join(config_arguments.input_pcl_folder, "{}_dim".format(config_arguments.output_dim), "*")
    args = "rm {} {}".format(sdv_dir, output_dir)
    subprocess.call(args, shell=True)

    # Compute new Input Parameterizations
    for i in range(0,len(point_cloud_files)):
        args = "./3DSmoothNet -f " + point_cloud_files[i] + " -k " + keypoints_files[i] + " -o " + config_arguments.evaluate_input_folder + " -r " + str(voxel_grid) + " -n " + str(n_voxels) + " -h " + str(gaussian_width)
        subprocess.call(args, shell=True)

    print('Input parametrization complete. Start inference')

    # Run the inference as shell 
    print("\n-----------------------")
    # Build the model and optimizer    
    smooth_net = network.NetworkBuilder(config_arguments)
    # Select the run mode
    if config_arguments.run_mode == "train":
        config_arguments.run_mode = "test"
    print('Run mode "{}" selected.'.format(config_arguments.run_mode))

    # Evaluate the network
    smooth_net.test()

    print('Inference completed perform nearest neighbor search and registration')

    
if __name__ == "__main__":
    # Parse configuration
    config_arguments, unparsed_arguments = config.get_config()

    # If we have unparsed arguments, print usage and exit
    if len(unparsed_arguments) > 0:
        config.print_usage()
        exit(1)

    main(config_arguments)

    arg_config_file = os.path.join(config_arguments.input_pcl_folder, "args_config.json")
    with open(arg_config_file, "w") as f:
        json.dump(vars(config_arguments), f, indent=2)

    print("Saved to args_config.json")

