# config.py ---
#
# Filename: config.py
# Description: Based on config file from https://github.com/vcg-uvic/learned-correspondence-release
# Author:  Zan Gojcic, Caifa Zhou
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet 
# Created: 03.04.2019
# Version: 1.0	

# Code:

# Import python dependencies
import argparse


arg_lists = []
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Keypoints
keypoint_arg = add_argument_group("Keypoints")
keypoint_arg.add_argument("--input_pcl_folder", type=str, default="./data/bunny/")
keypoint_arg.add_argument("--salient_factor", type=float, default=6.0)
keypoint_arg.add_argument("--nonmax_factor", type=float, default=2.0)
keypoint_arg.add_argument("--gamma_21", type=float, default=0.975)
keypoint_arg.add_argument("--gamma_32", type=float, default=0.975)
keypoint_arg.add_argument("--min_neighbors", type=int, default=5)

# -----------------------------------------------------------------------------
# TEASER++
teaserpp_arg = add_argument_group("Teaser++")
teaserpp_arg.add_argument("--cbar2", type=float, default=1, 
                          help ="square of maximum ratio between noise and noise bound (set to 1 by default).")
teaserpp_arg.add_argument("--noise_bound", type=float, default=0.03,
                          help="maximum bound on noise (depends on the data, default to 0.03).")
teaserpp_arg.add_argument("--estimate_scaling", action='store_true', 
                          help=" true if scale needs to be estimated, false otherwise (default to true).")
teaserpp_arg.add_argument("--rotation_estimation_algorithm", type=int, default=0,
                          help="0 for GNC-TLS, 1 for FGR (default to 0).")
teaserpp_arg.add_argument("--rotation_gnc_factor", type=float, default=1.4,
                          help= ("factor for increasing/decreasing the GNC function control parameter (default to 1.4):" 
                            "for GNC-TLS method: it’s multiplied on the GNC control parameter." 
                            "for FGR method: it’s divided on the GNC control parameter." ))
teaserpp_arg.add_argument("--rotation_max_iterations", type=int, default=100,
                          help="maximum iterations for the GNC-TLS/FGR loop (default to 100).")
teaserpp_arg.add_argument("--rotation_cost_threshold", type=float, default=0.005,
                          help="cost threshold for FGR termination (default to 0.005).")
teaserpp_arg.add_argument("--visualize", action='store_true')
# -----------------------------------------------------------------------------
# Input parametrization
voxel_arg = add_argument_group("Parametrization")
voxel_arg.add_argument("--voxel_grid", type=float, default=0.15,
                       help="Half size of the voxel grid in the unit of the point cloud. Defaults to 0.15.")
voxel_arg.add_argument("--n_voxels", type=int, default=16,
                       help="Number of voxels in a side of the grid. Whole grid is nxnxn. Defaults to 16.")
voxel_arg.add_argument("--gaussian_width", type=float, default=1.75,
                       help="Width of the Gaussia kernel used for smoothing. Defaults to 1.75.")


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument("--run_mode", type=str, default="test",
                     help='run_mode')
net_arg.add_argument('--input_dim', type=int, default=4096,
                     help='the dimension of the input features')
net_arg.add_argument('--output_dim', type=int, default=32,
                     help='the dimension of the learned local descriptor')
net_arg.add_argument('--log_path', type=str, default='./logs',
                     help='path to the directory with the tensorboard logs')
# -----------------------------------------------------------------------------
# Test
test_arg = add_argument_group("Evaluate")
test_arg.add_argument("--evaluate_input_folder", type=str, default="./data/evaluate/input_data/",
                          help='prefix for the input folder locations')
test_arg.add_argument("--evaluate_output_folder", type=str, default="./data/evaluate/output_data/",
                          help='prefix for the output folder locations')
test_arg.add_argument('--evaluation_batch_size', type=int, default=1000,
                          help='the number of examples for each iteration of inference')
test_arg.add_argument('--saved_model_dir', type=str, default='./models/',
                     help='the directory of the pre-trained model')
test_arg.add_argument('--saved_model_evaluate', type=str, default='3DSmoothNet',
                     help='file name of the model to load')
# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--input_data_folder", type=str, default="./data/train/input_data/",
                      help='prefix for the input folder locations')
train_arg.add_argument("--output_data_folder", type=str, default="./data/train/output_data/",
                      help='prefix for the output folder locations')
train_arg.add_argument('--max_steps', type=int, default=20000000,
                       help='maximum number of training iterations')
train_arg.add_argument('--max_epochs', type=int, default=20,
                       help='maximum number of training epochs')
train_arg.add_argument('--batch_size', type=int, default=256,
                       help='the number of training examples for each iteration')
train_arg.add_argument('--learning_rate', type=float, default=1e-3,
                       help='the initial learning rate')
train_arg.add_argument('--evaluate_rate', type=int, default=100,
                       help='frequency of evaluation')
train_arg.add_argument('--save_model_rate', type=int, default=1000,
                       help='the frequency of saving the check point')
train_arg.add_argument('--save_accuracy_rate', type=int, default=500,
                       help='the frequency of saving the training and validation accuracy')
train_arg.add_argument('--margin', type=str, default='soft',
                       help='the margin fucntion used for the loss')
train_arg.add_argument('--dropout_rate', type=float, default=0.7,
                       help='the keep probability')
train_arg.add_argument('--resume_flag', type=int, default=0,
                       help='the flag for training using the pre-trained model (1) or not (0)')
train_arg.add_argument('--decay_rate', type=float, default=0.95,
                       help='the rate of exponential learning rate decaying')
train_arg.add_argument('--decay_step', type=int, default=5000,
                       help='the frequency of exponential learning rate decaying')
train_arg.add_argument('--shuffle_size_TFRecords', type=int, default=5000,
                       help='the shuffle buffer size of the TFRecords')
train_arg.add_argument('--training_data_folder', type=str, default="./data/train/trainingData3DMatch",
                       help='location of the training data files')
train_arg.add_argument('--pretrained_model', type=str, default="./models/32_dim/3DSmoothNet_32_dim.ckpt",
                       help='pretrained model which will be used if resume is activared')

# Validation
valid_arg = add_argument_group("Validation")
train_arg.add_argument('--validation_data_folder', type=str, default="./data/validation/validationData3Dmatch/",
                       help='location of the validation data files')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()

#
# config.py ends here
