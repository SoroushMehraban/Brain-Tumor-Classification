from torch.utils.data import Dataset
import os
import cv2

TUMOR_TO_CLASS = {
    'glioma_tumor': 0,
    'meningioma_tumor': 1,
    'no_tumor': 2,
    'pituitary_tumor': 3
}


class BrainTumorDataset(Dataset):
    def __init__(self, images: list):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image_path = self.images[ix]
        image = self.read_image(image_path)

        tumor_type = image_path.split(os.sep)[-1].split('-')[0]
        return image, TUMOR_TO_CLASS[tumor_type]

    @staticmethod
    def read_image(image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def test_dataset():
    images = glob('final_dataset/*')
    dataset = BrainTumorDataset(images)
    print(images[0])
    print(dataset[0])
    print('----------')
    print(images[-1])
    print(dataset[-1])


if __name__ == '__main__':
    from glob import glob

    test_dataset()
