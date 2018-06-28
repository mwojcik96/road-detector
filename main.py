import glob
import random

import cv2
import numpy as np


def preprocess(image):
    return image


def postprocess(image):
    return image


def compare(pred, real):
    return pred == real


def cropper(img_size, crop_size):
    x = random.randint(0, img_size - crop_size)
    y = random.randint(0, img_size - crop_size)
    return x, x + crop_size, y, y + crop_size


def file_list(in_directory, out_directory):
    input_list = [file for file in glob.glob(in_directory + "/*.tiff")]
    output_list = [out_directory + "/" + file.split("/")[-1][:-1] for file in input_list]
    return input_list, output_list

batch_per_img = 100
images_for_batch = 10
img_size = 1500
input_size = 224  # FCN input element width and height
input_directory = './input_train'
output_directory = './output_train'

if __name__ == "__main__":
    input_list, output_list = file_list(input_directory, output_directory)

    # choose images_for_batch of images from training set
    samples = random.sample(range(0, len(input_list) - 1), images_for_batch)
    input_sample = [input_list[item] for item in samples]
    output_sample = [output_list[item] for item in samples]

    for input_img_path, output_img_path in zip(input_sample, output_sample):
        print(input_img_path)
        # read image input and output
        image_in = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        image_in = cv2.normalize(image_in.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_out = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)
        image_out = cv2.normalize(image_out.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        for i in range(batch_per_img):
            x, xx, y, yy = cropper(img_size, input_size)
            cv2.imshow("img", image_in[x:xx, y:yy])
            cv2.imshow("img2", image_out[x:xx, y:yy])
            cv2.waitKey(0)

