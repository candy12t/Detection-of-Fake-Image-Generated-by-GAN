import pickle

import numpy as np

from feature_extraction import get_feature
from image_process import resize_img


def predict_image(img):
    model = _load_model()

    X = _load_image(img)
    predict = model.predict(X)
    proba = model.predict_proba(X)

    proba = proba[0][predict[0]] * 100
    proba = round(proba, 2)
    predict = '偽物' if predict[0] == 0 else '本物'
    return predict, proba


def _load_model():
    model = './../model/dnn_202101301141_2.pickle'
    with open(model, mode='rb') as fp:
        loaded_model = pickle.load(fp)
    return loaded_model


def _load_image(img):
    img = resize_img(img)
    img = [get_feature(img)]
    X = np.array(img)
    X = X / 255
    X = X.reshape(len(X), -1)
    return X


if __name__ == '__main__':
    img = '../celeba_man/fake/9992_split.jpg'
    predict, proba = predict_image(img)
    print(predict)
    print(proba)
