import glob
import cv2
import numpy as np


def output_list(true_directory, pred_directory):
    t_list = [file for file in glob.glob(true_directory + "/*.tif")]
    p_list = [pred_directory + "/" + file.split("/")[-1][:-4] + ".png" for file in t_list]
    return t_list, p_list


def prepare_filter_list():
    odds = [3, 5, 7]
    thresholding = [0.5]
    element = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
    sigma = [0.5, 1, 2]
    filter_list = []
    # erode
    for mask in element:
        for x in odds:
            for y in odds:
                filter_list.append(('erode', mask, x, y))
    # dilate
    for mask in element:
        for x in odds:
            for y in odds:
                filter_list.append(('dilate', mask, x, y))
    # blur
    for x in odds:
        for y in odds:
            filter_list.append(('blur', x, y))
    # gaussian_blur
    for x in odds:
        for y in odds:
            for sigm_x in sigma:
                for sigm_y in sigma:
                    filter_list.append(('gaussian_blur', x, y, sigm_x, sigm_y))
    # median_blur
    #for x in odds:
    #    filter_list.append(('median_blur', x))
    # threshold
    for x in thresholding:
        filter_list.append(('threshold', x))

    return filter_list


def prepare_threshold_list():
    thresholding = [0.5]
    threshold_list = []
    # threshold
    for x in thresholding:
        threshold_list.append(('threshold', x))

    return threshold_list


def filter_applier(img, filter_to_aply):
    if filter_to_aply[0] == 'erode':
        return cv2.erode(img, cv2.getStructuringElement(shape=filter_to_aply[1], ksize=(filter_to_aply[2], filter_to_aply[3])))
    elif filter_to_aply[0] == 'dilate':
        return cv2.dilate(img, cv2.getStructuringElement(shape=filter_to_aply[1], ksize=(filter_to_aply[2], filter_to_aply[3])))
    elif filter_to_aply[0] == 'blur':
        return cv2.blur(img, (filter_to_aply[1], filter_to_aply[2]))
    elif filter_to_aply[0] == 'gaussian_blur':
        return cv2.GaussianBlur(img, (filter_to_aply[1], filter_to_aply[2]), sigmaX=filter_to_aply[3], sigmaY=filter_to_aply[4])
    #elif filter_to_aply[0] == 'median_blur':
    #    return cv2.medianBlur(img, filter_to_aply[1])
    elif filter_to_aply[0] == 'threshold':
        _, score = cv2.threshold(img, filter_to_aply[1], 1.0, cv2.THRESH_BINARY)
        return score
    else:
        return img


def decode_number(number, filter_size, threshold_size):
    filter_list = []
    threshold_number = number % threshold_size
    rest_number = int(number/threshold_size)
    while rest_number >= filter_size:
        filter_list.append(rest_number % filter_size)
        rest_number = int(rest_number/filter_size)
    filter_list.append(rest_number)
    return filter_list, threshold_number


def iou(img_true, img_pred):
    img_pred = img_pred >= 0.5
    return np.sum(np.bitwise_and(img_true, img_pred)) / np.sum(np.bitwise_or(img_true, img_pred))

if __name__ == "__main__":
    true_list, pred_list = output_list("./output_test", "./output_pred")

    t_img_list = []
    p_img_list = []
    for id, (t_img, p_img) in enumerate(zip(true_list, pred_list)):
        current_scores = []
        image_true = cv2.imread(t_img, cv2.IMREAD_GRAYSCALE)
        image_true = cv2.normalize(image_true.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_true = image_true >= 0.5
        image_pred = cv2.imread(p_img, cv2.IMREAD_GRAYSCALE)
        image_pred = cv2.normalize(image_pred.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        t_img_list.append(image_true)
        p_img_list.append(image_pred)

    # THRESHOLD -> OPEN/CLOSE
    for oper in [cv2.MORPH_OPEN, cv2.MORPH_CLOSE]:
        for threshold_1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for size in [3, 5, 7, 9, 11, 13, 15]:
                for kern in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                    #for threshold_2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        current_scores = []
                        for img_t, img_p in zip(t_img_list, p_img_list):
                            img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                            img_p = cv2.morphologyEx(img_p, oper, cv2.getStructuringElement(kern, (size, size)))
                            #img_p = cv2.threshold(img_p, threshold_2, 1.0, cv2.THRESH_BINARY)[1]
                            current_scores.append(iou(img_t, img_p))
                        if oper == cv2.MORPH_OPEN:
                            print("T-O", "T:", threshold_1, "O_size:", size, "O_kern:", kern, "AVG:",
                                  sum(current_scores) / len(current_scores), "MED:", np.median(current_scores), "MIN:",
                                  min(current_scores), "MAX:", max(current_scores))
                        else:
                            print("T-C", "T:", threshold_1, "C_size:", size, "C_kern:", kern, "AVG:",
                                  sum(current_scores) / len(current_scores), "MED:", np.median(current_scores), "MIN:",
                                  min(current_scores), "MAX:", max(current_scores))

    # OPEN/CLOSE -> THRESHOLD
    for oper in [cv2.MORPH_OPEN, cv2.MORPH_CLOSE]:
        for threshold_2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for size in [3, 5, 7, 9, 11, 13, 15]:
                for kern in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                    # for threshold_2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    current_scores = []
                    for img_t, img_p in zip(t_img_list, p_img_list):
                        # img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                        img_p = cv2.morphologyEx(img_p, oper, cv2.getStructuringElement(kern, (size, size)))
                        img_p = cv2.threshold(img_p, threshold_2, 1.0, cv2.THRESH_BINARY)[1]
                        current_scores.append(iou(img_t, img_p))
                    if oper == cv2.MORPH_OPEN:
                        print("O-T", "T:", threshold_2, "O_size:", size, "O_kern:", kern, "AVG:",
                              sum(current_scores) / len(current_scores), "MED:", np.median(current_scores), "MIN:",
                              min(current_scores), "MAX:", max(current_scores))
                    else:
                        print("C-T", "T:", threshold_2, "C_size:", size, "C_kern:", kern, "AVG:",
                              sum(current_scores) / len(current_scores), "MED:", np.median(current_scores), "MIN:",
                              min(current_scores), "MAX:", max(current_scores))

    # Thershold -> DED
    for threshold_1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for size_1 in [3, 5, 7, 9, 11, 13, 15]:
            for kern_1 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                for size_2 in [3, 5, 7, 9, 11, 13, 15]:
                    for kern_2 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                        for size_3 in [3, 5, 7, 9, 11, 13, 15]:
                            for kern_3 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                                current_scores = []
                                for img_t, img_p in zip(t_img_list, p_img_list):
                                    img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_1, (size_1, size_1)))
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_2, (size_2, size_2)))
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_3, (size_3, size_3)))
                                    current_scores.append(iou(img_t, img_p))
                                print("T-DED", "T:", threshold_1, "S1:", size_1, "K1:", kern_1, "S2:", size_2,
                                      "K2:", kern_2, "S3:", size_3, "K3:", kern_3, "AVG:",
                                      sum(current_scores) / len(current_scores), "MED:", np.median(current_scores),
                                      "MIN:", min(current_scores), "MAX:", max(current_scores))

    # Thershold -> EDE
    for threshold_1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for size_1 in [3, 5, 7, 9, 11, 13, 15]:
            for kern_1 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                for size_2 in [3, 5, 7, 9, 11, 13, 15]:
                    for kern_2 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                        for size_3 in [3, 5, 7, 9, 11, 13, 15]:
                            for kern_3 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                                current_scores = []
                                for img_t, img_p in zip(t_img_list, p_img_list):
                                    img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_1, (size_1, size_1)))
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_1, (size_2, size_2)))
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_1, (size_3, size_3)))
                                    current_scores.append(iou(img_t, img_p))
                                print("T-EDE", "T:", threshold_1, "S1:", size_1, "K1:", kern_1, "S2:", size_2,
                                      "K2:", kern_2, "S3:", size_3, "K3:", kern_3, "AVG:",
                                      sum(current_scores) / len(current_scores), "MED:", np.median(current_scores),
                                      "MIN:", min(current_scores), "MAX:", max(current_scores))

    # DED -> Thershold
    for threshold_1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for size_1 in [3, 5, 7, 9, 11, 13, 15]:
            for kern_1 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                for size_2 in [3, 5, 7, 9, 11, 13, 15]:
                    for kern_2 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                        for size_3 in [3, 5, 7, 9, 11, 13, 15]:
                            for kern_3 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                                current_scores = []
                                for img_t, img_p in zip(t_img_list, p_img_list):
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_1, (size_1, size_1)))
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_2, (size_2, size_2)))
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_3, (size_3, size_3)))
                                    img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                                    current_scores.append(iou(img_t, img_p))
                                print("T-DED", "T:", threshold_1, "S1:", size_1, "K1:", kern_1, "S2:", size_2,
                                      "K2:", kern_2, "S3:", size_3, "K3:", kern_3, "AVG:",
                                      sum(current_scores) / len(current_scores), "MED:", np.median(current_scores),
                                      "MIN:", min(current_scores), "MAX:", max(current_scores))

    # EDE -> Thershold
    for threshold_1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for size_1 in [3, 5, 7, 9, 11, 13, 15]:
            for kern_1 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                for size_2 in [3, 5, 7, 9, 11, 13, 15]:
                    for kern_2 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                        for size_3 in [3, 5, 7, 9, 11, 13, 15]:
                            for kern_3 in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                                current_scores = []
                                for img_t, img_p in zip(t_img_list, p_img_list):
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_1, (size_1, size_1)))
                                    img_p = cv2.dilate(img_p, cv2.getStructuringElement(kern_1, (size_2, size_2)))
                                    img_p = cv2.erode(img_p, cv2.getStructuringElement(kern_1, (size_3, size_3)))
                                    img_p = cv2.threshold(img_p, threshold_1, 1.0, cv2.THRESH_BINARY)[1]
                                    current_scores.append(iou(img_t, img_p))
                                print("T-EDE", "T:", threshold_1, "S1:", size_1, "K1:", kern_1, "S2:", size_2,
                                      "K2:", kern_2, "S3:", size_3, "K3:", kern_3, "AVG:",
                                      sum(current_scores) / len(current_scores), "MED:", np.median(current_scores),
                                      "MIN:", min(current_scores), "MAX:", max(current_scores))


                    #cv2.imshow("a", image_true)
    #cv2.imshow("b", image_pred)
    #cv2.waitKey(0)