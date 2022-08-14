import os
from pathlib import Path
from glob import glob
import imutils
import cv2

TUMOR_TYPES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def get_cropped_coords(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)

    thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    brain_contour = max(contours, key=cv2.contourArea)

    x_min = tuple(brain_contour[brain_contour[:, :, 0].argmin()][0])[0]
    x_max = tuple(brain_contour[brain_contour[:, :, 0].argmax()][0])[0]
    y_min = tuple(brain_contour[brain_contour[:, :, 1].argmin()][0])[1]
    y_max = tuple(brain_contour[brain_contour[:, :, 1].argmax()][0])[1]

    return x_min, y_min, x_max, y_max


def create_folders_if_not_exist(out_path):
    directory = os.sep.join(out_path.split(os.sep)[:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)


def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def store_image(cropped_image, image_path):
    out_path = image_path.replace(f"..{os.sep}", "")
    create_folders_if_not_exist(out_path)
    cv2.imwrite(out_path, cropped_image)


def crop_contours(mode):
    assert mode in ['Training', 'Testing']

    for tumor_type in TUMOR_TYPES:
        image_paths = glob(f"..{os.sep}dataset{os.sep}{mode}{os.sep}{tumor_type}{os.sep}*")
        for image_path in image_paths:
            image = read_image(image_path)
            x_min, y_min, x_max, y_max = get_cropped_coords(image)
            cropped_image = image[y_min:y_max, x_min:x_max]
            store_image(cropped_image, image_path)


def main():
    crop_contours('Training')
    crop_contours('Testing')


if __name__ == '__main__':
    main()
