import argparse

MODEL = 'PSPNet'  # PSPNet, DeepLab, RefineNet
# RESTORE_FROM = './pretrained_models/pretrain_pspnet_150000.pth'
# RESTORE_FROM = './pretrained_models/dannet14000_51mIoU.pth'
RESTORE_FROM = './pretrained_models/dannet10000_47-43mIoU.pth'

BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4

SET = 'train'
DATA_DIRECTORY = '/home/jerry/xxz/panoptic-deepalb_qysun/panoptic-deeplab-master/datasets/cityscapes'
DATA_LIST_PATH = './dataset/lists/cityscapes_train.txt'
INPUT_SIZE = '512'
DATA_DIRECTORY_TARGET = '/home/jerry/xxz/DANNet/dataset/darkzurich'
DATA_LIST_PATH_TARGET = './dataset/lists/zurich_dn_pair_train.csv'
INPUT_SIZE_TARGET = '960'

NUM_CLASSES = 19
IGNORE_LABEL = 255

#LEARNING_RATE = 2.5e-4
LEARNING_RATE = 0.001
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4

NUM_STEPS = 10000
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'+MODEL
STD = 0.05
# PSEUDO_MODE = 'generate_previously'
PSEUDO_MODE = 'generate_with_training'
USE_FORGE_LABELS = False
AUX_LOSS_WEIGHT = 0.1

ALL_PRETRAINED = False
# LIGHTNET_RESTORE_FROM = './pretrained_models/dannet_light14000_51mIoU.pth'
# D1_RESTORE_FROM = './pretrained_models/dannet_d1_14000_51mIoU.pth'
# D2_RESTORE_FROM = './pretrained_models/dannet_d2_14000_51mIoU.pth'
LIGHTNET_RESTORE_FROM = './pretrained_models/dannet_light10000_47-43mIoU.pth'
D1_RESTORE_FROM = './pretrained_models/dannet_d1_10000_47-43mIoU.pth'
D2_RESTORE_FROM = './pretrained_models/dannet_d2_10000_47-43mIoU.pth'


def get_arguments():
    parser = argparse.ArgumentParser(description="DANNet")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=int, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--pseudo_mode", type=str, default=PSEUDO_MODE)
    parser.add_argument("--use_forge_labels", type=bool, default=USE_FORGE_LABELS)
    parser.add_argument("--aux_loss_weight", type=float, default=AUX_LOSS_WEIGHT)
    parser.add_argument("--all-pretrained", type=bool, default=ALL_PRETRAINED)
    parser.add_argument("--lightnet-restore-from", type=str, default=LIGHTNET_RESTORE_FROM)
    parser.add_argument("--d1-restore-from", type=str, default=D1_RESTORE_FROM)
    parser.add_argument("--d2-restore-from", type=str, default=D2_RESTORE_FROM)
    return parser.parse_args()
