""""
TP1 - Multimédia
David Leitão [2019223148]
Rodrigo Machado [2019218299]
Rui Costa [2019224237]
"""

from typing import Tuple
import numpy as np
from matplotlib import image, colors, pyplot as plt
from scipy import ndimage, fftpack as fft
import cv2
from pprint import pprint


#----- Packaged Encoder/Decoder -----#

def encoder(path: str, sampling: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    img = image.imread(path)
    shape = img.shape
    img = padding(img)
    r, g, b = sepRGB(img)
    y, cb, cr = ycbcr(r, g ,b)
    if sampling != (4,4,4):
        cb, cr = subsampler((cb, cr), sampling)
    return y, cb, cr, shape


def decoder(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], shape: tuple) -> np.ndarray:
    y, cb, cr = ycbcr
    cb, cr = upsampler(cb, cr, y.shape)
    img = rgb(y, cb, cr)
    img = unpadding(img, shape)
    return img

#----- RGB Colormaps -----#

RED = colors.LinearSegmentedColormap.from_list('cmap', [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 256)
GREEN = colors.LinearSegmentedColormap.from_list('cmap', [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], 256)
BLUE = colors.LinearSegmentedColormap.from_list('cmap', [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], 256)

def viewImage(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    title = kwargs.get("title", None)
    block = kwargs.get("block", True)
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    figure = plt.figure(num=title)
    figure.figimage(image, resize=True, cmap=cmap)
    plt.show(block=block)


def showImage(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    plt.axis("off")
    plt.imshow(image, cmap=cmap)


#----- RGB Channel splitting and joining -----#

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


#----- Padding/Unpadding -----#

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
    r, g, b = sepRGB(img)

    if verticalPadding > 0:
        r = np.vstack((r, np.tile(r[-1, :], (verticalPadding, 1))))
        g = np.vstack((g, np.tile(g[-1, :], (verticalPadding, 1))))
        b = np.vstack((b, np.tile(b[-1, :], (verticalPadding, 1))))
    if horizontalPadding > 0:
        r = np.hstack((r, np.tile(r[:, -1], (horizontalPadding, 1)).transpose()))
        g = np.hstack((g, np.tile(g[:, -1], (horizontalPadding, 1)).transpose()))
        b = np.hstack((b, np.tile(b[:, -1], (horizontalPadding, 1)).transpose()))

    return joinRGB(r, g, b)


def unpadding(img: np.ndarray, shape: np.shape) -> np.ndarray:
    return img[:shape[0], :shape[1], :]


#----- Colorspace conversions -----#

YCbCr = np.array(
    [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
)


def ycbcr(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = YCbCr[0, 0] * r + YCbCr[0, 1] * g + YCbCr[0, 2] * b
    Cb = YCbCr[1, 0] * r + YCbCr[1, 1] * g + YCbCr[1, 2] * b + 128
    Cr = YCbCr[2, 0] * r + YCbCr[2, 1] * g + YCbCr[2, 2] * b + 128
    return Y, Cb, Cr


def rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    table = np.linalg.inv(YCbCr)
    Cb = Cb - 128
    Cr = Cr - 128
    r = table[0, 0] * Y + table[0, 1] * Cb + table[0, 2] * Cr
    g = table[1, 0] * Y + table[1, 1] * Cb + table[1, 2] * Cr
    b = table[2, 0] * Y + table[2, 1] * Cb + table[2, 2] * Cr
    img = joinRGB(r,g,b)
    img = np.round(img)
    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


def viewYCbCr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray):
    plt.subplot(131)
    showImage(y, cmap="gray")
    plt.subplot(132)
    showImage(cb, cmap="gray")
    plt.subplot(133)
    showImage(cr, cmap="gray")
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)
    # plt.show()


#----- Chroma resampling -----#

def subsampler(chroma: tuple, ratio: tuple) -> Tuple[np.ndarray, np.ndarray]:
    if ratio == (4,4,4): return
    cb, cr = chroma
    cbRatio = ratio[1]/ratio[0]
    horizontal = False
    if ratio[2] == 0:
        crRatio = cbRatio
        horizontal = True
    else:
        crRatio = ratio[2]/ratio[0]
    
    cbStep = int(1/cbRatio)
    crStep = int(1/crRatio)
    cb = cb[:, ::cbStep]
    cr = cr[:, ::crStep]
    if horizontal:
        cb = cb[::cbStep, :]
        cr = cr[::crStep, :]

    return (cb, cr)
    

def upsampler(cb: np.ndarray, cr: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    cbZoom = (shape[0] / cb.shape[0], shape[1] / cb.shape[1])
    crZoom = (shape[0] / cr.shape[0], shape[1] / cr.shape[1])
    print(cbZoom, crZoom)
    cb = ndimage.zoom(cb, cbZoom)
    cr = ndimage.zoom(cr, crZoom)
    return cb, cr


#----- Discrete Cosine Transform -----#

def dct(X: np.ndarray) -> np.ndarray:
    return fft.dct(fft.dct(X, norm="ortho").T, norm="ortho").T


def idct(X: np.ndarray) -> np.ndarray:
    return fft.idct(fft.idct(X, norm="ortho").T, norm="ortho").T
    

def viewDct(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> None:
    x1log = np.log(np.abs(x1) + 0.0001)
    x2log = np.log(np.abs(x2) + 0.0001)
    x3log = np.log(np.abs(x3) + 0.0001)
    viewYCbCr(x1log, x2log, x3log)


#----- Main -----#

def main():
    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = barn

    plt.figure("YCbCr")
    Y, Cb, Cr, shape = encoder(file, (4, 4, 4))
    Cb, Cr = subsampler((Cb,Cr), (4,1,0))
    viewYCbCr(Y, Cb, Cr)

    plt.figure("DCT")
    Y_dct = dct(Y)
    Cb_dct = dct(Cb)
    Cr_dct = dct(Cr)
    viewDct(Y_dct, Cb_dct, Cr_dct)

    plt.figure("IDCT")
    Y_inv = idct(Y_dct)
    Cb_inv = idct(Cb_dct)
    Cr_inv = idct(Cr_dct)
    viewYCbCr(Y_inv, Cb_inv, Cr_inv)

    decoded = decoder((Y_inv,Cb_inv,Cr_inv), shape)
    plt.figure("Compressed")
    showImage(decoded)
    plt.show()


if __name__ == "__main__":
    main()
