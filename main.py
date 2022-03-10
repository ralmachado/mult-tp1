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
from PIL import Image

# ----- Packaged Encoder/Decoder -----#


def encoder(path: str, sampling: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    img = image.imread(path)
    shape = img.shape
    r, g, b = sepRGB(img)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    if sampling != (4, 4, 4):
        cb, cr = cvSubsampler((cb, cr), sampling)
    return y, cb, cr, shape


def decoder(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], shape: tuple) -> np.ndarray:
    y, cb, cr = ycbcr
    cb, cr = cvUpsampler(cb, cr, y.shape)
    img = rgb(y, cb, cr)
    img = unpadding(img, shape)
    return img


# ----- RGB Colormaps -----#

RED = colors.LinearSegmentedColormap.from_list(
    "cmap", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 256
)
GREEN = colors.LinearSegmentedColormap.from_list(
    "cmap", [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], 256
)
BLUE = colors.LinearSegmentedColormap.from_list(
    "cmap", [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], 256
)


def showImage(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    title = kwargs.get("title", None)
    block = kwargs.get("block", True)
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    figure = plt.figure(num=title)
    figure.figimage(image, resize=True, cmap=cmap)
    plt.show(block=block)


def viewImage(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    plt.axis("off")
    plt.imshow(image, cmap=cmap)


def viewYCbCr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> None:
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)
    plt.subplot(1, 3, 1)
    viewImage(y, cmap="gray")
    plt.subplot(1, 3, 2)
    viewImage(cb, cmap="gray")
    plt.subplot(1, 3, 3)
    viewImage(cr, cmap="gray")

def viewRGB(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> None:
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)
    plt.subplot(1, 3, 1)
    viewImage(r, cmap=RED)
    plt.subplot(1, 3, 2)
    viewImage(g, cmap=GREEN)
    plt.subplot(1, 3, 3)
    viewImage(b, cmap=BLUE)


# ----- RGB Channel splitting and joining -----#


def sepRGB(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim > 2:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        return r, g, b
    else:
        raise Exception("Image has a single channel")


def joinRGB(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    if r.shape != g.shape or g.shape != b.shape or r.shape != b.shape:
        raise Exception("Shape mismatch")

    shape = (r.shape[0], r.shape[1], 3)
    rgb = np.empty(shape, dtype=r.dtype)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


# ----- Padding/Unpadding -----#


def padding(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if r.shape != g.shape or g.shape != b.shape or r.shape != b.shape:
        raise Exception("Shape mismatch")

    height, width = r.shape

    modY = height % 16
    modX = width % 16
    verticalPadding = 16 - modY if modY != 0 else 0
    horizontalPadding = 16 - modX if modX != 0 else 0

    if verticalPadding > 0:
        r = np.vstack((r, np.tile(r[-1, :], (verticalPadding, 1))))
        g = np.vstack((g, np.tile(g[-1, :], (verticalPadding, 1))))
        b = np.vstack((b, np.tile(b[-1, :], (verticalPadding, 1))))
    if horizontalPadding > 0:
        r = np.hstack((r, np.tile(r[:, -1], (horizontalPadding, 1)).transpose()))
        g = np.hstack((g, np.tile(g[:, -1], (horizontalPadding, 1)).transpose()))
        b = np.hstack((b, np.tile(b[:, -1], (horizontalPadding, 1)).transpose()))

    return r, g, b


def unpadding(img: np.ndarray, shape: np.shape) -> np.ndarray:
    return img[: shape[0], : shape[1], :]


# ----- Colorspace conversions -----#

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
    img = joinRGB(r, g, b)
    img = np.round(img)
    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


#----- Chroma Resampling -----#


def cvSubsampler(chroma: Tuple[np.ndarray, np.ndarray], ratio: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Chroma downsampling using cv2."""

    if ratio == (4, 4, 4):
        return cb, cr
    cb, cr = chroma
    horizontal = False
    cbRatio = ratio[1] / ratio[0]
    if ratio[2] == 0:
        crRatio = cbRatio
        horizontal = True
    else:
        crRatio = ratio[2] / ratio[0]

    if horizontal:
        cb = cv2.resize(cb, dsize=None, fx=cbRatio, fy=cbRatio, interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr, dsize=None, fx=crRatio, fy=crRatio, interpolation=cv2.INTER_AREA)
    else:
        cb = cv2.resize(cb, dsize=None, fy=cbRatio, interpolation=cv2.INTER_AREA)
        cr = cv2.resize(cr, dsize=None, fy=crRatio, interpolation=cv2.INTER_AREA)

    return cb, cr


def cvUpsampler(cb: np.ndarray, cr: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Chroma upsampling using cv2."""

    size = shape[::-1]

    cb = cv2.resize(cb, size, interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(cr, size, interpolation=cv2.INTER_CUBIC)

    return cb, cr


# ----- Deprecated chroma resampling -----#


def subsampler(chroma: tuple, ratio: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deprecated.

    Old chroma subsampling function. Kept for historical reasons.
    """

    if ratio == (4, 4, 4):
        return
    cb, cr = chroma
    cbStep = int(ratio[0] / ratio[1])
    horizontal = False
    if ratio[2] == 0:
        crStep = cbStep
        horizontal = True
    else:
        crStep = int(ratio[0] / ratio[2])

    cb = cb[:, ::cbStep]
    cr = cr[:, ::crStep]
    if horizontal:
        cb = cb[::cbStep, :]
        cr = cr[::crStep, :]

    return (cb, cr)


def upsampler(cb: np.ndarray, cr: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deprecated.

    Old chroma upsampling function using scipy. Kept for historical reasons.
    """

    cbZoom = (shape[0] / cb.shape[0], shape[1] / cb.shape[1])
    crZoom = (shape[0] / cr.shape[0], shape[1] / cr.shape[1])
    cb = ndimage.zoom(cb, cbZoom)
    cr = ndimage.zoom(cr, crZoom)
    return cb, cr


# ----- Discrete Cosine Transform -----#


def dct(X: np.ndarray) -> np.ndarray:
    return fft.dct(fft.dct(X, norm="ortho").T, norm="ortho").T


def idct(X: np.ndarray) -> np.ndarray:
    return fft.idct(fft.idct(X, norm="ortho").T, norm="ortho").T


def viewDct(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> None:
    x1log = np.log(np.abs(x1) + 0.0001)
    x2log = np.log(np.abs(x2) + 0.0001)
    x3log = np.log(np.abs(x3) + 0.0001)
    viewYCbCr(x1log, x2log, x3log)


def blockDct(x: np.ndarray, size: int = 8) -> np.ndarray:
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = dct(x[i:i+size, j:j+size])
    return newImg


def blockIdct(x: np.ndarray, size: int = 8) -> np.ndarray:
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
    return newImg


# ----- Quantization ----- #

QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])


def quantize(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], qf: int = 75) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, cb, cr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255

    qy = np.empty(y.shape)
    qcb = np.empty(cb.shape)
    qcr = np.empty(cr.shape)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            qy[i:i+8, j:j+8] = y[i:i+8, j:j+8] / QsY
    np.round(qy)

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            qcb[i:i+8, j:j+8] = cb[i:i+8, j:j+8] / QsC
    np.round(qcb)

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            qcr[i:i+8, j:j+8] = cr[i:i+8, j:j+8] / QsC
    np.round(qcr)

    return qy.astype(np.uint8), qcb.astype(np.uint8), qcr.astype(np.uint8)

def iQuantize(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], qf: int = 75) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qy, qcb, qcr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255

    y = np.empty(y.shape)
    cb = np.empty(cb.shape)
    cr = np.empty(cr.shape)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            y[i:i+8, j:j+8] = qy[i:i+8, j:j+8] * QsY

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            cb[i:i+8, j:j+8] = qcb[i:i+8, j:j+8] * QsC

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            cr[i:i+8, j:j+8] = qcr[i:i+8, j:j+8] * QsC

    return y, cb, cr


# ----- Main ----- #


def oldmain():
    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = barn

    plt.figure("YCbCr")
    Y, Cb, Cr, shape = encoder(file, (4, 4, 4))
    Cb, Cr = subsampler((Cb, Cr), (4, 1, 0))
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

    decoded = decoder((Y_inv, Cb_inv, Cr_inv), shape)
    plt.figure("Compressed")
    viewImage(decoded)
    plt.show()


def main():
    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = barn
    pillow = Image.open(file)
    img = np.array(pillow)
    originalShape = img.shape

    plt.figure("Original Image")
    viewImage(img)

    # Separate RGB channels
    r, g, b = sepRGB(img)
    plt.figure("RGB Channels")
    viewRGB(r, g, b)

    # Add padding to make image sides multiples of 16
    r, g, b = padding(r, g, b)
    plt.figure("Padding")
    viewRGB(r, g, b)

    # RGB to YCbCr colorspace conversion
    y, cb, cr = ycbcr(r, g, b)
    plt.figure("RGB to YCbCr")
    viewYCbCr(y, cb, cr)

    # Chroma subsampling
    plt.figure("Chroma Subsampling")
    ratio = (4, 2, 0)
    cb, cr = cvSubsampler((cb,cr), ratio)
    viewYCbCr(y, cb, cr)

    # Whole-image DCT
    # plt.figure("Whole-image DCT")
    # y = dct(y)
    # cb = dct(cb)
    # cr = dct(cr)
    # viewDct(y, cb, cr)

    # Whole-image inverse DCT
    # plt.figure("Whole-image Inverse DCT")
    # y = idct(y)
    # cb = idct(cb)
    # cr = idct(cr)
    # viewYCbCr(y, cb, cr)

    # Block DCT
    block = 8
    plt.figure("Block DCT")
    y = blockDct(y, size=block)
    cb = blockDct(cb, size=block)
    cr = blockDct(cr, size=block)
    viewDct(y, cb, cr)

    # Whole-image inverse DCT
    plt.figure("Block Inverse DCT")
    y = blockIdct(y, size=block)
    cb = blockIdct(cb, size=block)
    cr = blockIdct(cr, size=block)
    viewYCbCr(y, cb, cr)

    # Chroma upsampling
    plt.figure("Upsampling")
    cb, cr = cvUpsampler(cb, cr, y.shape)
    viewYCbCr(y, cb, cr)

    # YCbCr to RGB colorspace conversion
    plt.figure("YCbCr to RGB")
    img = rgb(y, cb, cr)
    r, g, b = sepRGB(img) # Separation is only done here for YCbCr to RGB conversion
    viewRGB(r, g, b)

    # Remove padding
    plt.figure("Reconstructed Image")
    img = unpadding(img, originalShape)
    viewImage(img)

    plt.show()


if __name__ == "__main__":
    main()
