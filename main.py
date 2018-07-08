import glob
import os
import random

import cv2
import keras.backend as K
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from matplotlib import pyplot

from model import FullyConvolutionalNetwork


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
    return x, y


def cropper(img_size, crop_size):
    x = random.randint(0, img_size - crop_size)
    y = random.randint(0, img_size - crop_size)
    return x, x + crop_size, y, y + crop_size


def file_list(in_directory, out_directory):
    input_list = [file for file in glob.glob(in_directory + "/*.tiff")]
    output_list = [out_directory + "/" + file.split("/")[-1][:-1] for file in input_list]
    return input_list, output_list


def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True), axis=-1)

def intersection_over_union(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_t = K.greater_equal(y_true_f, 0.5)
    y_p = K.greater_equal(y_pred_f, 0.5)
    union1 = [i for i, j in zip(y_true_f, y_pred_f) if i or j]
    union2 = [j for i, j in zip(y_true_f, y_pred_f) if i or j]
    intersection = [i for i, j in zip(y_true_f, y_pred_f) if i and j]
    unionAll = union1 + union2 + intersection
    return (np.sum(intersection) + smooth) / float(np.sum(unionAll) + smooth)


batch_per_img = 32
images_for_batch = 5
img_size = 1500
input_size = 128  # FCN input element width and height
input_directory = './input_train'
output_directory = './output_train'
weight_decay = 0 # 1e-3
save_path = './model'
training = False
epochs = 500
epochs_per_batch = 1
batch_size = 32
final_test = True

def dice_coefff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefff(y_true, y_pred)


def loss_func(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def lr_scheduler(epoch):
    lr = lr_base * ((1 - float(epoch)) ** 0.9)
    print('lr: %f' % lr)
    return lr


if __name__ == "__main__":

    model = FullyConvolutionalNetwork(input_shape=(input_size, input_size, 3), weight_decay=weight_decay, classes=1)
    print(model.summary())
    lr_base = 0.01 * (float(batch_size) / 16)
    optimizer = SGD(lr=lr_base, momentum=0.9)

    model.compile(loss=dice_loss, optimizer=optimizer, metrics=[dice_coefff])

    if training:
        model_path = os.path.join(save_path, "model.json")
        # save model structure
        f = open(model_path, 'w')
        model_json = model.to_json()
        f.write(model_json)
        f.close()

        print(model.summary())

        #for xxx, yyy in zip(x, y):
        #    cv2.imshow("x", xxx)
        #    cv2.imshow("y", yyy)
        #    cv2.waitKey(0)
        scheduler = LearningRateScheduler(lr_scheduler)
        callbacks = [scheduler]
        for i in range(epochs):
            x, y = prepare_batch(input_directory, output_directory, images_for_batch, batch_per_img)

            model.fit(x, y, batch_size=batch_size, epochs=epochs_per_batch, callbacks=callbacks)

            model.save_weights(save_path + '/model_sigm.hdf5')

    elif not final_test:
        for layer in model.layers:
            print(layer.get_weights())
            print("--------------")
        x, y = prepare_batch('./input_train', './output_train', images_for_batch, batch_per_img)
        print(x.shape)
        print(y.shape)
        y_pred = model.predict(x)
        for xx, yy, zz in zip(x, y, y_pred):
            print("COEFF:", (2. * np.sum(np.multiply(yy, zz)) + 1.0) / (np.sum(yy) + np.sum(zz) + 1.0))
            minn, maxx = np.percentile(np.squeeze(zz), [2, 98])
            print(minn, maxx)
            cv2.imshow("y_pred_tt", np.squeeze(zz))
            zz = (zz - minn) / (maxx - minn)
            cv2.imshow("x", np.squeeze(xx))
            cv2.imshow("y_true", np.squeeze(yy))
            cv2.imshow("y_pred", np.squeeze(zz))
            cv2.imshow("y_pred_t", np.where(np.squeeze(zz) > 0.5, 1.0, 0.0))
            cv2.waitKey(0)

    else:
        split = 49
        input_list, output_list = file_list('./input_test', './output_test')
        for in_img, out_img in zip(input_list, output_list):
            print('./output_pred/' + in_img.split('/')[-1][:-5] + '.png')
            image_in = cv2.imread(in_img, cv2.IMREAD_COLOR)
            image_in = cv2.normalize(image_in.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            image_out = cv2.imread(out_img, cv2.IMREAD_GRAYSCALE)
            image_out = cv2.normalize(image_out.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

            pred_label = np.zeros(shape=(img_size, img_size), dtype=float)
            pred_counter = np.zeros(shape=(img_size, img_size), dtype=float)
            for i in range(int(1 + (img_size - input_size)/split)):
                img_batch = np.empty(shape=(0, input_size, input_size, 3), dtype=np.float)
                print(i)
                for j in range(int(1 + (img_size - input_size)/split)):
                    #img_batch = np.append(img_batch, [image_in[i*split:i*split+input_size, j*split:j*split+input_size, :]], axis=0)
                    img_batch = np.append(img_batch, np.array([image_in[i*split:i*split+input_size, j*split:j*split+input_size]]), axis=0)

                y_predict = model.predict(img_batch)

                for j in range(int(1 + (img_size - input_size)/split)):
                    y_pred = np.squeeze(y_predict[j])
                    y_pred_cast = np.pad(y_pred, pad_width=((i*split, img_size - i*split - input_size), (j*split, img_size - j*split - input_size)), mode='constant', constant_values=(0,0))
                    y1, y2, x1, x2 = i*split, img_size - i*split - input_size, j*split, img_size - j*split - input_size
                    y_pred_count = np.pad(np.ones((input_size, input_size), dtype=float), pad_width=[(y1, y2), (x1, x2)], mode='constant', constant_values=(0,0))
                    pred_label = np.add(pred_label, y_pred_cast)
                    pred_counter = np.add(pred_counter, y_pred_count)

            print(pred_counter[120:160, 120:160])

            score = np.divide(pred_label, pred_counter)

            print("IOU:", np.sum(np.logical_and(score, image_out)) / np.sum(np.logical_or(score, image_out)))
            pyplot.imsave('./output_pred/' + in_img.split('/')[-1][:-5] + '.png', score, cmap='gray')

            #score = cv2.morphologyEx(score, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
            #cv2.imshow("y_true", image_out)
            #cv2.imshow("y_pred", score)
            #cv2.waitKey(0)