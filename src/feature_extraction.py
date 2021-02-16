import re
from pathlib import Path

import numpy as np

from image_process import max_min_filter

WITHOUT_NUMBER = re.compile(r"[^0-9]")  # 数値以外(正規表現)
NUM_BINS = 256
RANGE = (0, 256)


def get_feature(img, feature="edge_hist"):
    gray = img.convert("L")  # グレイスーケル化
    rgb = np.array(img).astype(np.float)
    gray = np.array(gray).astype(np.float)

    if feature == "hist":
        data = _get_histogram(rgb)
    elif feature == "edge":
        data = _get_edge(gray)
    elif feature == "edge_hist":
        hist = _get_histogram(rgb)
        edge = _get_edge(gray)
        data = np.append(hist, edge)
    return data


def get_label(f):
    """
    f = '../celeba_man/fake/9992_split.jpg'
    get_label(f) -> 'fake'
    """
    return Path(f).parts[-2]


# ファイル名に含まれる数値を取得
# 分類に失敗した(不正解)データを記録するため
def get_file_number(f):
    """
    f = '../celeba_man/fake/9992_split.jpg'
    get_file_number(f) -> 9992
    """
    filename_without_ext = Path(f).stem
    file_number = WITHOUT_NUMBER.sub("", filename_without_ext)
    return int(file_number)


def _get_histogram(img):
    img, _ = np.histogram(img.reshape(-1), bins=NUM_BINS, range=RANGE)
    return img


def _get_edge(img):
    img = max_min_filter(img)
    img = _get_histogram(img)
    return img
