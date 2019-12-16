"""
config file
Taken from : https://github.com/rbgirshick/yacs
"""
from yacs.config import CfgNode as CN


_C = CN()

_C.NAME = "default"

_C.SYSTEM = CN()

# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 1  # :'(
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 8
_C.SYSTEM.SAVED_MODEL_DIR = 'checkpoints'
_C.SYSTEM.PRETRAINED_MODEL = ''
_C.SYSTEM.LOG_DIR = "logs"

_C.INPUT = CN()

# PATH of the json pair (containing relative path path of images and label)

# path of training directory
_C.INPUT.TRAINING_DIRECTORY = "../../3DMatch/training/"

# shape of the depth images
_C.INPUT.INPUT_SHAPE = (640, 480)

# size of the patch
_C.INPUT.NUM_POINT = 2000
_C.INPUT.RADIUS = 0.2
_C.INPUT.IN_FEATURES_DIM = 1
_C.INPUT.POINTS_DIM = 3
_C.INPUT.NUM_CLASSES = 40
_C.INPUT.MAX_NUM_NEIGHBORS = 17

_C.NETWORK = CN()
_C.NETWORK.FIRST_DIM = 64
_C.NETWORK.BATCH_NUM = 25
_C.NETWORK.ARCHITECTURE = ["simple", "simple", "simple", "simple", "global_average"]
_C.NETWORK.KP_EXTENT = 1.0
_C.NETWORK.KP_INFLUENCE = 'linear'
_C.NETWORK.FIXED_KERNEL_POINTS = 'center'
_C.NETWORK.CONVOLUTION_MODE = 'sum'
_C.NETWORK.DENSITY_PARAMETER = 5.0
_C.NETWORK.FEATURES_DIM = [32, 64, 64, 128, 128]
_C.NETWORK.OUTPUT_DIM = 32
_C.NETWORK.FIRST_SUBSAMPLING_DL = 0.01
_C.NETWORK.NUM_KERNEL_POINTS = 15
_C.NETWORK.USE_BATCH_NORM = True
_C.NETWORK.BATCH_NORM_MOMENTUM = 0.99
# Behavior of convolutions in ('closest', 'sum')
# Decide if you sum all kernel point influences, or if you only take the influence of the closest KP


# Fixed points in the kernel : 'none', 'center' or 'verticals'
_C.AUGMENT = CN()
_C.AUGMENT.AUGMENT_NOISE = 0.005
_C.AUGMENT.AUGMENT_ROTATION = 'all'
_C.AUGMENT.AUGMENT_SCALE_ANISOTROPIC = False
_C.AUGMENT.AUGMENT_SCALE_MIN = 1.0
_C.AUGMENT.AUGMENT_SCALE_MAX = 1.0
_C.AUGMENT.IS_AUGMENT = False
_C.AUGMENT.AUGMENT_SYMMETRIES = [False, False, False]
_C.AUGMENT.AUGMENT_OCCLUSION = 'none'

_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 0.01
_C.TRAIN.MOMENTUM = 0.9

_C.TRAIN.LR_DECAY = 0.99
_C.TRAIN.USE_BATCH_NORM = True
_C.TRAIN.BATCH_NORM_MOMENTUM = 0.99
_C.TRAIN.EPOCH = 10
_C.TRAIN.GRAD_CLIP_NORM = 100.0
_C.TRAIN.WEIGHTS_DECAY = 0.0
_C.TRAIN.MODULATED = False
_C.TRAIN.MAX_EPOCH = 5
_C.TRAIN.OFFSETS_LOSS = 'permissive'
_C.TRAIN.OFFSETS_DECAY = 1e-2
_C.TRAIN.MARGIN = 'soft'
_C.TRAIN.SNAPSHOT_GAP = 5
_C.TRAIN.SAVING = True
_C.TRAIN.SAVING_KP = True
_C.TRAIN.LOG_INTERVAL = 10


def get_cfg_defaults():
  """
  Get a yacs CfgNode object with default values for my_project.
  """
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
