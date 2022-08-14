import os
from glob import glob
from pathlib import Path
import shutil

TUMOR_TYPES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
MODES = ['Training', 'Testing']


def create_new_file_structure():
    Path(f'..{os.sep}final_dataset').mkdir(parents=True, exist_ok=True)
    for tumor_type in TUMOR_TYPES:
        tumor_images = gather_tumor_images(tumor_type)
        copy_to_new_location(tumor_images, tumor_type)


def copy_to_new_location(tumor_images, tumor_type):
    for idx, tumor_image_path in enumerate(tumor_images, 1):
        shutil.copyfile(tumor_image_path, f'..{os.sep}final_dataset{os.sep}{tumor_type}-{idx}.jpg')


def gather_tumor_images(tumor_type):
    tumor_images = []
    for mode in MODES:
        tumor_images += glob(f'dataset{os.sep}{mode}{os.sep}{tumor_type}{os.sep}*')
    return tumor_images


def main():
    create_new_file_structure()


if __name__ == '__main__':
    main()
