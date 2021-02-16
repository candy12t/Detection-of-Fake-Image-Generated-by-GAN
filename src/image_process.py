from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from PIL import Image

NUM_BINS = 256
RWIDTH = 0.8
RANGE = (0, 256)


# matplolibを日本語に対応
mpl.style.use("default")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",
    "Meirio",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "Noto Sans CJK JP",
]


# エッジ検出(MAX-MINフィルタ)
def max_min_filter(img, K_size=3):
    if len(img.shape) == 3:
        H, W, C = img.shape

        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2, 3), dtype=np.float)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                for c in range(3):
                    out[pad+y, pad+x, c] = \
                            np.max(tmp[y:y+K_size, x:x+K_size, c]) \
                            - np.min(tmp[y:y+K_size, x:x+K_size, c])
        out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    else:
        H, W = img.shape

        pad = K_size // 2
        out = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
        out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                out[pad+y, pad+x] = \
                        np.max(tmp[y:y+K_size, x:x+K_size]) \
                        - np.min(tmp[y:y+K_size, x:x+K_size])
        out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


def resize_img(input_img, size=64):
    img = Image.open(input_img)
    img = img.resize((size, size), Image.LANCZOS)
    return img


def get_filename_without_extension(f):
    """
    f = '../celeba_man/fake/9992_split.jpg'
    get_file_number(f) -> '9992_split'
    """
    return Path(f).stem


def main(input_path, mode):
    path = Path(input_path)
    if path.is_dir():
        for f in path.glob("*.jpg"):
            _save_processed_img(f)
    else:
        _save_processed_img(path)


def _save_processed_img(f):
    if mode == "histogram":
        _save_histogram_img(f)
    elif mode == "edge":
        _save_edge_img(f)
    elif mode == "resize":
        _save_resize_img(f)


def _save_histogram_img(input_img):
    img = Image.open(input_img)
    img = np.array(img).astype(float)
    filename_without_ext = get_filename_without_extension(input_img)

    ax = plt.subplot()
    ax.hist(img.reshape(-1), bins=NUM_BINS, rwidth=RWIDTH, range=RANGE)
    ax.set_xlabel("輝度値")
    ax.set_ylabel("輝度の出現回数")
    plt.savefig(f"./histogram_{filename_without_ext}.png")
    plt.close()


def _save_edge_img(input_img):
    img = Image.open(input_img).convert("L")
    img = np.array(img).astype(float)
    filename_without_ext = get_filename_without_extension(input_img)

    edge_img = max_min_filter(img)
    edge_img = Image.fromarray(edge_img)
    edge_img.save(f"./max_min_{filename_without_ext}.jpg")


def _save_resize_img(input_img, size=64):
    img = resize_img(input_img, size)
    filename_without_ext = get_filename_without_extension(input_img)
    img.save(f"./resize_{size}x{size}_{filename_without_ext}.jpg", quality=95)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str,
                        required=True, help="input image path")
    parser.add_argument("-m", "--mode", default="histogram",
                        type=str, choices=["histogram", "edge", "resize"],
                        help="""
                             select image process model
                             {'histogram','edge','resize'},
                             default='histogram'
                             """)
    args = parser.parse_args()
    input_path = args.input_path
    mode = args.mode

    main(input_path, mode)
