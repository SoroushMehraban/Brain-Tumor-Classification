import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import *
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AlexNet(class_numbers=4).to(DEVICE)

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

LEARNING_RATE = 1e-3
EPOCHS = 5000
BATCH_SIZE = 64

RUNTIME_INFO_FILE = "<PATH TO RUNTIME INFO HERE>"
CHECKPOINT_FILE = "<PATH TO CHECKPOINT HERE>"

RANDOM_SEED = 5
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