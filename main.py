""""
TP1 - Multimédia
David Leitão [2019223148]
Rodrigo Machado [2019218299]
Rui Costa [2019224237]
"""

import math
from pprint import pprint
from typing import Union, Tuple
from unittest import result
import numpy as np
from matplotlib import image, pyplot as plt

"""
Semana 1
    Done: 1, 2, 3.1, 3.4
    To Implement: 3.2, 3.3, 4, 5

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


def sepRGB(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    height = img.shape[0]
    width = img.shape[1]

    # The resulting sizes are even since 16 is even
    modY = height % 16
    modX = width % 16
    verticalPadding = 16 - modY if modY != 0 else 0
    horizontalPadding = 16 - modX if modX != 0 else 0

    # We assume that all images are RGB
    r,g,b = sepRGB(img)

    if verticalPadding > 0:
        r = np.vstack((r, np.tile(r[-1, :], (verticalPadding, 1))))
        g = np.vstack((g, np.tile(g[-1, :], (verticalPadding, 1))))
        b = np.vstack((b, np.tile(b[-1, :], (verticalPadding, 1))))
    if horizontalPadding > 0:
        r = np.hstack((r, np.tile(r[:, -1], (horizontalPadding, 1)).transpose()))
        g = np.hstack((g, np.tile(g[:, -1], (horizontalPadding, 1)).transpose()))
        b = np.hstack((b, np.tile(b[:, -1], (horizontalPadding, 1)).transpose()))
    
    return joinRGB(r,g,b)

def unpadding(img: np.ndarray) -> np.ndarray:
    pass


YCbCr = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])


def ycbcr(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = YCbCr[0,0] * r + YCbCr[0,1] * g + YCbCr[0,2] * b
    Cb = YCbCr[1,0] * r + YCbCr[1,1] * g + YCbCr[1,2] * b - 128
    Cr = YCbCr[2,0] * r + YCbCr[2,1] * g + YCbCr[2,2] * b - 128
    return Y, Cb, Cr

def rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.linalg.inv(YCbCr)
    Cb = Cb - 128
    Cr = Cr - 128
    r = table[0,0] * Y + table[0,1] * Cb + table[0,2] * Cr
    g = table[1,0] * Y + table[1,1] * Cb + table[1,2] * Cr
    b = table[2,0] * Y + table[2,1] * Cb + table[2,2] * Cr
    return r, g, b


if __name__ == "__main__":
    viewOriginal = False
    viewChannels = False
    viewJoined = False
    viewPadded = False

    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = peppers

    img = image.imread(file)
    if viewOriginal:
        viewImage(img, title="Original")

    r, g, b = sepRGB(img)
    if viewChannels:
        viewImage(r, block=False, title="Red Channel", cmap="Reds")
        viewImage(g, block=False, title="Green Channel", cmap="Greens")
        viewImage(b, title="Blue Channel", cmap="Blues")

    rgbImage = joinRGB(r, g, b)
    if viewJoined:
        viewImage(rgbImage, title="Joined Channels")

    paddedImg = padding(img)
    if viewPadded:
        viewImage(paddedImg, title="Padded")

    y, cb, cr = ycbcr(r,g,b)
    r,g,b = rgb(y,cb,cr)
    img = joinRGB(r,g,b)
    viewImage(img)
    # viewImage(y, block=False, title="Y Channel")
    # viewImage(cb, block=False, title="Cb Channel")
    # viewImage(cr, title="Cr Channel")

