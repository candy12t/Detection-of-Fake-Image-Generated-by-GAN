import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import feature_extraction as fe


def get_nearly_real_fake_dataset_lists():
    dataset_lists = [
        "../celeba_man/real_500.txt",
        "../celeba_man/nearly_real_fake_500.txt",
    ]
    return dataset_lists


# 画像を読み込んで特徴量を抽出して数値化
def load_dataset(dataset_lists, feature="edge_hist"):
    data, labels, filenames = [], [], []

    for dataset_list in dataset_lists:
        for f in _read_dataset_list(dataset_list):
            img = Image.open(f)

            img = fe.get_feature(img, feature)
            label = fe.get_label(f)
            filename = fe.get_file_number(f)

            data.append(img)
            labels.append(label)
            filenames.append(filename)

    X, y, z = np.array(data), np.array(labels), np.array(filenames)
    return _normalize_dataset(X, y, z)


# *.txtに書かれた画像のパスを`\n`を削除してリストに格納
def _read_dataset_list(text_path):
    with open(text_path, mode="r", encoding="utf_8") as f:
        img_path_list = [s.strip() for s in f.readlines()]
        return img_path_list


# 正規化
def _normalize_dataset(X, y, z):
    X = X / 255
    X = X.reshape(len(X), -1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X, y, z
