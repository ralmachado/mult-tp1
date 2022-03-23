""""
TP1 - Multimédia
David Leitão [2019223148]
Rodrigo Machado [2019218299]
Rui Costa [2019224237]
"""

import cv2
from matplotlib import image, colors, pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, fftpack as fft
from typing import Tuple


# ----- Packaged Encoder/Decoder -----#


def encoder(path: str, sampling: tuple, qf: int = 75) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    img = image.imread(path)
    shape = img.shape
    r, g, b = sepRGB(img)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    yOriginal = np.copy(y)
    if sampling != (4, 4, 4):
        cb, cr = subsampler((cb, cr), sampling)
    y = blockDct(y)
    cb = blockDct(cb)
    cr = blockDct(cr)
    y, cb, cr = quantize((y,cb,cr), qf)
    y = DPCM(y)
    cb = DPCM(cb)
    cr = DPCM(cr)

    return y, cb, cr, shape, yOriginal


def decoder(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], shape: tuple, qf: int = 75) -> np.ndarray:
    y, cb, cr = ycbcr
    y = iDPCM(y)
    cb = iDPCM(cb)
    cr = iDPCM(cr)
    y,cb,cr = iQuantize((y,cb,cr), qf)
    y = blockIdct(y)
    cb = blockIdct(cb)
    cr = blockIdct(cr)
    cb, cr = upsampler(cb, cr, y.shape)
    img = rgb(y, cb, cr)
    img = unpadding(img, shape)
    return img, y


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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
    plt.imshow(image, cmap=cmap)


def viewYCbCr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> None:
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)
    plt.subplot(131)
    plt.title("Y Channel")
    viewImage(y, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Cb Channel")
    viewImage(cb, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Cr Channel")
    viewImage(cr, cmap="gray")

def viewRGB(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> None:
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01)
    plt.subplot(1, 3, 1)
    plt.title("Red Channel")
    viewImage(r, cmap=RED)
    plt.subplot(1, 3, 2)
    plt.title("Green Channel")
    viewImage(g, cmap=GREEN)
    plt.subplot(1, 3, 3)
    plt.title("Blue Channel")
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
        r = np.hstack((r, np.tile(r[:, -1], (horizontalPadding, 1)).T))
        g = np.hstack((g, np.tile(g[:, -1], (horizontalPadding, 1)).T))
        b = np.hstack((b, np.tile(b[:, -1], (horizontalPadding, 1)).T))

    return r, g, b


def unpadding(img: np.ndarray, shape: np.shape) -> np.ndarray:
    return img[: shape[0], : shape[1], :]


def viewPadding(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    img = joinRGB(r, g, b)
    plt.title("Padding")
    viewImage(img)


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


def subsampler(chroma: Tuple[np.ndarray, np.ndarray], ratio: tuple) -> Tuple[np.ndarray, np.ndarray]:
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


def upsampler(cb: np.ndarray, cr: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Chroma upsampling using cv2."""

    size = shape[::-1]

    cb = cv2.resize(cb, size, interpolation=cv2.INTER_CUBIC)
    cr = cv2.resize(cr, size, interpolation=cv2.INTER_CUBIC)

    return cb, cr


# ----- Deprecated chroma resampling -----#


def oldSubsampler(chroma: tuple, ratio: tuple) -> Tuple[np.ndarray, np.ndarray]:
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


def oldUpsampler(cb: np.ndarray, cr: np.ndarray, shape: tuple) -> Tuple[np.ndarray, np.ndarray]:
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
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
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
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1

    qy = np.empty(y.shape, dtype=y.dtype)
    qcb = np.empty(cb.shape, dtype=cb.dtype)
    qcr = np.empty(cr.shape, dtype=cr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            qy[i:i+8, j:j+8] = y[i:i+8, j:j+8] / QsY
    qy = np.round(qy)

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            qcb[i:i+8, j:j+8] = cb[i:i+8, j:j+8] / QsC
    qcb = np.round(qcb)

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            qcr[i:i+8, j:j+8] = cr[i:i+8, j:j+8] / QsC
    qcr = np.round(qcr)

    return qy, qcb, qcr


def iQuantize(ycbcr: Tuple[np.ndarray, np.ndarray, np.ndarray], qf: int = 75) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qy, qcb, qcr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1

    y = np.empty(qy.shape, dtype=qy.dtype)
    cb = np.empty(qcb.shape, dtype=qcb.dtype)
    cr = np.empty(qcr.shape, dtype=qcr.dtype)

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


#-----DPCM-----#


def DPCM(x: np.ndarray) -> np.ndarray:
    dc0 = x[0,0]
    r,c = x.shape
    for i in range(0, r, 8):
        for j in range(0, c, 8):
            if i == 0 and j == 0:
                continue
            dc = x[i,j]
            diff = dc - dc0
            dc0 = dc
            x[i,j] = diff
    return x


def iDPCM(x:np.ndarray) -> np.ndarray:
    r,c =  x.shape
    dc0 = x[0, 0]
    for i in range(0, r, 8):
        for j in range(0, c, 8):
            if i == 0 and j == 0:
                continue
            dc = x[i,j]
            summ = dc + dc0
            dc0 = summ
            x[i,j] = summ
    return x


# ----- Metrics ----- #


# barn_mountains.bmp
# q = 75
# DS = 4:2:0
# MSE ~= 187
# SNR ~= 25


def MSE(x: np.ndarray, y: np.ndarray) -> np.float64:
    h, w = x[:,:,0].shape
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    coef = 1 / (h * w)
    return coef * (np.sum((x - y) ** 2))


def RMSE(mse: np.float64) -> np.float64:
    return mse ** (1 / 2)

def SNR(x: np.ndarray, mse: np.float64) -> np.float64:
    h, w = x[0].shape
    x = x.astype(np.float64)
    coef =  1 / (w * h)
    power = coef * np.sum(x ** 2)
    return 10 * np.log10(power / mse)


def PSNR(x: np.ndarray, mse: np.float64) -> np.float64:
    return 10 * np.log10((np.max(x) ** 2) / mse)


# ----- Main ----- #


def main():
    basePath = "imagens"
    peppers = f"{basePath}/peppers.bmp"
    logo = f"{basePath}/logo.bmp"
    barn = f"{basePath}/barn_mountains.bmp"

    file = barn
    qualityFactor = 75
    pillow = Image.open(file)
    img = np.array(pillow)
    originalShape = img.shape
    show = True

    if show:
        plt.figure("Original Image")
        viewImage(img)

    # Separate RGB channels
    r, g, b = sepRGB(img)
    if show:
        plt.figure("RGB Channels")
        viewRGB(r, g, b)

    # Add padding to make image sides multiples of 16
    r, g, b = padding(r, g, b)
    if show:
        plt.figure("Padding")
        viewRGB(r, g, b)

    # RGB to YCbCr colorspace conversion
    y, cb, cr = ycbcr(r, g, b)
    if show:
        plt.figure("RGB to YCbCr")
        viewYCbCr(y, cb, cr)

    # Chroma subsampling
    ratio = (4, 2, 0)
    cb, cr = subsampler((cb,cr), ratio)
    if show:
        plt.figure("Chroma Subsampling")
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
    block = 64
    dy = blockDct(y, size=block)
    dcb = blockDct(cb, size=block)
    dcr = blockDct(cr, size=block)
    if show:
        plt.figure("Block DCT (64x64)")
        viewDct(dy, dcb, dcr)
    block = 8
    y = blockDct(y, size=block)
    cb = blockDct(cb, size=block)
    cr = blockDct(cr, size=block)
    if show:
        plt.figure("Block DCT (8x8)")
        viewDct(y, cb, cr)

    # Quantization
    y, cb, cr = quantize((y,cb,cr), qualityFactor)
    plt.figure("Quantization")
    viewDct(y, cb, cr)

    # DPCM
    y = DPCM(y)
    cb = DPCM(cb)
    cr = DPCM(cr)
    if show:
        plt.figure("DPCM")
        viewDct(y, cb, cr)

    # Inverse DPCM
    y = iDPCM(y)
    cb = iDPCM(cb)
    cr = iDPCM(cr)
    if show:
        plt.figure("Inverse DPCM")
        viewDct(y, cb, cr)

    # Inverse quantization
    y, cb, cr = iQuantize((y, cb, cr), qualityFactor)
    # if show:
    plt.figure("Inverse Quantization")
    viewDct(y, cb, cr)

    # Inverse Block DCT
    y = blockIdct(y, size=block)
    cb = blockIdct(cb, size=block)
    cr = blockIdct(cr, size=block)
    if show:
        plt.figure("Block Inverse DCT")
        viewYCbCr(y, cb, cr)

    # Chroma upsampling
    cb, cr = upsampler(cb, cr, y.shape)
    if show:
        plt.figure("Upsampling")
        viewYCbCr(y, cb, cr)

    # YCbCr to RGB colorspace conversion
    img = rgb(y, cb, cr)
    r, g, b = sepRGB(img) # Separation is only done here for YCbCr to RGB conversion
    if show:
        plt.figure("YCbCr to RGB")
        viewRGB(r, g, b)

    # Remove padding
    img = unpadding(img, originalShape)
    plt.figure("Reconstructed Image")
    viewImage(img)

    plt.show()


def metrics(filepath: str, qf: int = 75, show: bool = True, metrics: bool = True) -> np.ndarray:
    original = np.array(Image.open(filepath))
    y, cb, cr, shape, yOriginal = encoder(filepath, (4,2,0), qf=qf)
    compressed, yReconstructed = decoder((y,cb,cr), shape, qf=qf)
    diff = np.absolute(yOriginal - yReconstructed)
    if show:
        plt.figure("Compressed")
        viewImage(compressed)
        plt.figure("Difference")
        viewImage(diff, cmap="gray")
    mse = MSE(original, compressed)
    rmse = RMSE(mse)
    snr = SNR(original, mse)
    psnr = PSNR(original, mse)
    if metrics:
        print(f"MSE: {mse:.3f}\nRMSE: {rmse:.3f}")
        print(f"SNR: {snr:.3f} dB\nPSNR: {psnr:.3f} dB")
    plt.show()
    return compressed, {'mse': mse, 'rmse': rmse, 'snr': snr, 'psnr': psnr}


def codec(filepath: str, qf: int = 75):
    compressed = metrics(filepath, qf, False, False)
    plt.figure("Compressed image")
    viewImage(compressed)
    plt.show()


if __name__ == "__main__":
    # main()
    metrics("imagens/barn_mountains.bmp", qf=75)
