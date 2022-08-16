import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import *
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AlexNet(class_numbers=4, in_channels=1).to(DEVICE)

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

TRAIN_RATIO = 0.75  # 75 %
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.15
RANDOM_SEED = 5
assert TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO == 1

LEARNING_RATE = 1e-3
SCHEDULER_STEP1 = 0.5  # 50% of epochs
SCHEDULER_STEP2 = 0.8
assert SCHEDULER_STEP2 > SCHEDULER_STEP1

BETA1 = 0.9
BETA2 = 0.999
EPOCHS = 300
BATCH_SIZE = 64

RUNTIME_INFO_FILE = "<PATH TO RUNTIME INFO HERE>"
CHECKPOINT_FILE = "<PATH TO CHECKPOINT HERE>"

PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True

transform = A.Compose(
    [
        A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
        A.Normalize(mean=0, std=1, max_pixel_value=255, ),
        ToTensorV2(),
    ],
)
