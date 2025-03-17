import argparse

# validation set path
# DATA_DIRECTORY = '/path/to/Dark_Zurich_val_anon/rgb_anon/val'
# DATA_LIST_PATH = './dataset/lists/zurich_val.txt'
DATA_DIRECTORY = '/home/jerry/xxz/DANNet/dataset/NighttimeDrivingTest/leftImg8bit/'
DATA_LIST_PATH = './dataset/lists/night-time-driving_val.txt'

# test set path
# DATA_DIRECTORY = '/path/to/public_data_2/rgb_anon'
# DATA_LIST_PATH = './dataset/lists/zurich_test.txt'

IGNORE_LABEL = 255
NUM_CLASSES = 19
SET = 'val'

MODEL = 'PSPNet'
# RESTORE_FROM = './trained_models/dannet_psp.pth'
# RESTORE_FROM_LIGHT = './trained_models/dannet_psp_light.pth'

RESTORE_FROM = './snapshots/PSPNet/dannet2000.pth'
RESTORE_FROM_LIGHT = './snapshots/PSPNet/dannet_light2000.pth'

# RESTORE_FROM = './snapshots/PSPNet_pseudoupdate_51mIoU/dannet14000.pth' #mIoU 51.5
# RESTORE_FROM_LIGHT = './snapshots/PSPNet_pseudoupdate_51mIoU/dannet_light14000.pth'
# RESTORE_FROM = './pretrained_models/dannet_onlyGAN39.pth'
# RESTORE_FROM_LIGHT = './pretrained_models/dannet_light_onlyGAN39.pth'
# RESTORE_FROM = './snapshots/PSPNet_d2n_GAN/dannet6000.pth'
# RESTORE_FROM_LIGHT = './snapshots/PSPNet_d2n_GAN/dannet_light6000.pth'

SAVE_PATH = './result/dannet_'+MODEL
STD = 0.16
USE_CLASS_CENTER_WEIGHT = False

SAVE_PROB = False

#####是否使用lightM
USE_LIGHTM = False
if USE_LIGHTM:
    LIGHTM_C = 1

#PSPNet_d2n_GAN_Mix_convlightmask_phi4.0
DATASET = 'NighttimeDriving'
# DATASET = 'DarkZurich'
DATASET = 'NightCity'
if DATASET == 'DarkZurich':
    DATA_DIRECTORY = '/home/jerry/xxz/panoptic-deepalb_qysun/panoptic-deeplab-master/datasets/darkzurich/leftImg8bit/'
    DATA_LIST_PATH = './dataset/lists/zurich_test.txt'
if DATASET == 'NightCity':
    DATA_DIRECTORY = '/home/jerry/xxz/DANNet/dataset/NightCity/images/'
    DATA_LIST_PATH = './dataset/lists/nightcity_val.txt'
    SAVE_PATH = './result/PSPNet_nightcity'


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-light", type=str, default=RESTORE_FROM_LIGHT,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--use-class-center-weight", type=bool, default=USE_CLASS_CENTER_WEIGHT)
    parser.add_argument("--save_prob", type=bool, default=SAVE_PROB)
    parser.add_argument("--use_lightM", type=bool, default=USE_LIGHTM)
    parser.add_argument("--dataset", type=str, default=DATASET)
    if USE_LIGHTM:
        parser.add_argument("--lightM_C", type=int, default=LIGHTM_C)
    return parser.parse_args()