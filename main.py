def preprocess(image):
    return image


def predict(image):
    return image


def postprocess(image):
    return image


def roads(image):
    input_img = preprocess(image)
    output_img = predict(input_img)
    score_img = postprocess(output_img)
    return score_img


if __name__ == "__main__":
    roads()