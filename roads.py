import os

import numpy as np
import cv2
from model import FullyConvolutionalNetwork

img_size = 1500
input_size = 128


def predict(in_img, model):
    split = 49
    print('./output_pred/' + in_img.split('/')[-1][:-5] + '.png')
    image_in = cv2.imread(in_img, cv2.IMREAD_COLOR)
    image_in = cv2.normalize(image_in.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    pred_label = np.zeros(shape=(img_size, img_size), dtype=float)
    pred_counter = np.zeros(shape=(img_size, img_size), dtype=float)
    for i in range(int(1 + (img_size - input_size) / split)):
        img_batch = np.empty(shape=(0, input_size, input_size, 3), dtype=np.float)
        print(i)
        for j in range(int(1 + (img_size - input_size) / split)):
            img_batch = np.append(img_batch, np.array(
                [image_in[i * split:i * split + input_size, j * split:j * split + input_size]]), axis=0)

        y_predict = model.predict(img_batch)

        for j in range(int(1 + (img_size - input_size) / split)):
            y_pred = np.squeeze(y_predict[j])
            y_pred_cast = np.pad(y_pred, pad_width=(
                (i * split, img_size - i * split - input_size), (j * split, img_size - j * split - input_size)),
                                 mode='constant', constant_values=(0, 0))
            y1, y2, x1, x2 = i * split, img_size - i * split - input_size, j * split, img_size - j * split - input_size
            y_pred_count = np.pad(np.ones((input_size, input_size), dtype=float), pad_width=[(y1, y2), (x1, x2)],
                                  mode='constant', constant_values=(0, 0))
            pred_label = np.add(pred_label, y_pred_cast)
            pred_counter = np.add(pred_counter, y_pred_count)

        print(pred_counter[120:160, 120:160])

        score = np.divide(pred_label, pred_counter)


def roads(img):
    image_in = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    model = FullyConvolutionalNetwork()
    weights_path = os.path.expanduser(os.path.join('./model/model_sigm.hdf5'))
    model.load_weights(weights_path, by_name=True)
    predict(image_in, model)
