""""
TP1 - Multimédia
David Leitão [2019223148]
Rodrigo Machado [2019218299]
"""

import math
from pprint import pprint
from typing import Union
import numpy as np
from matplotlib import image, pyplot as plt

"""
Semana 1
    Done: 1, 2, 3.1, 3.4
    To Implement: 3.2, 3.3, 4, 5, ball hehe

    When converting YCbCr back to RGB, first round, then clamp values to [0, 255], then cast with astype(np.uint8)
    Yes daddy.
"""


def enconder():
    pass


def decoder():
    pass


def viewImage(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    title = kwargs.get("title", None)
    block = kwargs.get("block", True)
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    figure = plt.figure(num=title)
    figure.figimage(image, resize=True, cmap=cmap)
    plt.show(block=block)


def sepRGB(image: np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim > 2:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        return r, g, b
    else:
        raise Exception("image.ndim not bigger than 2")


def joinRGB(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    if r.shape != g.shape or g.shape != b.shape or r.shape != b.shape:
        raise Exception("Shape mismatch")

    shape = (r.shape[0], r.shape[1], 3)
    rgb = np.empty(shape, dtype=r.dtype)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def padding(img: np.ndarray) -> np.ndarray:
    if img.ndim < 2:
        raise Exception("img array is one dimensional")

    trueHeight = img.shape[0]
    trueWidth = img.shape[1]

    # The resulting sizes are even since 16 is even
    width = math.ceil(trueWidth / 16) * 16
    height = math.ceil(trueHeight / 16) * 16
    
    verticalPadding = height - trueHeight
    horizontalPadding = width - trueWidth

    # use hstack and vstack

    if img.ndim == 2:
        for _ in range(verticalPadding):
            img = np.vstack((img, img[-1, :]))
        for _ in range(horizontalPadding):
            img = np.hstack((img, img[:, -1]))
    else:
        pprint(img[:, -1, 0])
        for _ in range(verticalPadding):
            img = np.vstack((img, img[-1, :, :]))
        for _ in range(horizontalPadding):
            img = np.hstack((img, img[:, -1, :]))

    return img
    

def unpadding(img: np.ndarray) -> np.ndarray:
    pass


if __name__ == "__main__":
    viewOriginal = False
    viewChannels = False
    viewJoined = False

    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = barn

    img = image.imread(file)
    if viewOriginal == True:
        viewImage(img, title="Original")

    r, g, b = sepRGB(img)
    if viewChannels == True:
        viewImage(r, block=False, title="Red Channel", cmap="Reds")
        viewImage(g, block=False, title="Green Channel", cmap="Greens")
        viewImage(b, title="Blue Channel", cmap="Blues")

    rgb = joinRGB(r, g, b)
    if viewJoined == True:
        viewImage(rgb, title="Joined Channels")

    paddedImg = padding(img)
    viewImage(paddedImg, title="Padded")
