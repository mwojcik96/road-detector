import glob
import random
import os

import cv2
import numpy as np
from keras import losses
from keras.losses import binary_crossentropy
from keras.optimizers import SGD

from model import AtrousFCN_Vgg16_16s

def prepare_batch(input_dir, output_dir, img_for_batch=1, batch_p_img=1):
    input_list, output_list = file_list(input_dir, output_dir)

    # choose images_for_batch of images from training set
    samples = random.sample(range(0, len(input_list) - 1), img_for_batch)
    input_sample = [input_list[item] for item in samples]
    output_sample = [output_list[item] for item in samples]

    x = np.empty(shape=(0, input_size, input_size, 3), dtype=np.float)
    y = np.empty(shape=(0, input_size, input_size, 1), dtype=np.float)

    for input_img_path, output_img_path in zip(input_sample, output_sample):
        print(input_img_path)
        # read image input and output
        image_in = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        image_in = cv2.normalize(image_in.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_out = cv2.imread(output_img_path, cv2.IMREAD_GRAYSCALE)
        image_out = cv2.normalize(image_out.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_out = np.expand_dims(image_out, axis=2)

        for i in range(batch_p_img):
            xl, xr, yl, yr = cropper(img_size, input_size)
            x = np.append(x, np.array([image_in[xl:xr, yl:yr]]), axis=0)
            y = np.append(y, np.array([image_out[xl:xr, yl:yr]]), axis=0)
            #cv2.imshow("img", image_in[x:xx, y:yy])
            #cv2.imshow("img2", image_out[x:xx, y:yy])
            #cv2.waitKey(0)
    return x, y


def iou_lf(y_true, y_pred):
    print(y_true, y_pred)
    true_img = y_true > 0.5
    pred_img = y_pred > 0.5
    fin_img = true_img == pred_img

    print(fin_img)

    return 1 - cv2.countNonZero(fin_img) / (fin_img.shape[0] * fin_img.shape[1])


def cropper(img_size, crop_size):
    x = random.randint(0, img_size - crop_size)
    y = random.randint(0, img_size - crop_size)
    return x, x + crop_size, y, y + crop_size


def file_list(in_directory, out_directory):
    input_list = [file for file in glob.glob(in_directory + "/*.tiff")]
    output_list = [out_directory + "/" + file.split("/")[-1][:-1] for file in input_list]
    return input_list, output_list

batch_per_img = 100
images_for_batch = 1
img_size = 1500
input_size = 128  # FCN input element width and height
input_directory = './input_train'
output_directory = './output_train'
weight_decay = 1e-3
resume_training = False
save_path = './model'
training = True

if __name__ == "__main__":

    model = AtrousFCN_Vgg16_16s(input_shape=(input_size, input_size, 3), weight_decay=weight_decay, classes=1)
    print(model.summary())
    lr_base = 0.01 * (float(batch_per_img * images_for_batch) / 16)
    optimizer = SGD(lr=lr_base, momentum=0.9)

    model.compile(loss=losses.mean_squared_error, optimizer=optimizer)

    if training:
        if resume_training:
            checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
            model.load_weights(checkpoint_path, by_name=True)

        model_path = os.path.join(save_path, "model.json")
        # save model structure
        f = open(model_path, 'w')
        model_json = model.to_json()
        f.write(model_json)
        f.close()

        print(model.summary())

        x, y = prepare_batch(input_directory, output_directory, images_for_batch, batch_per_img)

        model.fit(x, y, batch_size=8, epochs=5)

        model.save_weights(save_path + '/model.hdf5')

    else:
        x, y = prepare_batch('./input_test', './output_test', images_for_batch, batch_per_img)
        y_pred = model.predict(x)
        for xx, yy, zz in zip(x, y, y_pred):
            minn, maxx = np.percentile(np.squeeze(zz), [2, 98])
            zz = (zz - minn) / (maxx - minn)
            cv2.imshow("x", np.squeeze(xx))
            cv2.imshow("y_true", np.squeeze(yy))
            cv2.imshow("y_pred", np.squeeze(zz))
            cv2.waitKey(0)