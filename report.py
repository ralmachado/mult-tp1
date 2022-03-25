from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from main import *


def colormodels(image: np.ndarray) -> None:
    r, g, b = sepRGB(image)
    plt.figure()
    viewRGB(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    plt.figure()
    viewYCbCr(y, cb, cr)

    
def subsampling(image: np.ndarray, ratio: tuple = (4, 2, 0)) -> None:
    r, g, b = sepRGB(image)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    cb, cr = subsampler((cb,cr), ratio)
    viewYCbCr(y, cb, cr)

def DCT(image: np.ndarray, ratio: tuple = (4, 2, 0), block: int = 0) -> None:
    r, g, b = sepRGB(image)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    cb, cr = subsampler((cb,cr), ratio)
    if block == 0:
        y = dct(y)
        cb = dct(cb)
        cr = dct(cr)
    else:
        y = blockDct(y, size=block)
        cb = blockDct(cb, size=block)
        cr = blockDct(cr, size=block)
    viewDct(y, cb, cr)

def quantization(image: np.ndarray, ratio: tuple = (4, 2, 0), qf: int = 75) -> None:
    r, g, b = sepRGB(image)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    cb, cr = subsampler((cb,cr), ratio)
    y = blockDct(y, size=8)
    cb = blockDct(cb, size=8)
    cr = blockDct(cr, size=8)
    y, cb, cr = quantize((y, cb, cr), qf=qf)
    viewDct(y, cb, cr)


def DPCM(image: np.ndarray, ratio: tuple = (4, 2, 0), qf: int = 75) -> None:
    r, g, b = sepRGB(image)
    r, g, b = padding(r, g, b)
    y, cb, cr = ycbcr(r, g, b)
    cb, cr = subsampler((cb,cr), ratio)
    y = blockDct(y, size=8)
    cb = blockDct(cb, size=8)
    cr = blockDct(cr, size=8)
    y, cb, cr = quantize((y, cb, cr), qf=qf)
    viewDct(y, cb, cr)
    y = dpcm(y)
    cb = dpcm(cb)
    cr = dpcm(cr)
    plt.figure()
    viewDct(y, cb, cr)


def verboseMetrics(original: np.ndarray, qf: int = 75, ratio: tuple = (4,2,0)):
    y, cb, cr, shape, yOriginal = encoder(original, ratio, qf=qf)
    compressed, yReconstructed = decoder((y,cb,cr), shape, qf=qf)
    diff = np.absolute(yOriginal - yReconstructed)
    plt.figure()
    plt.subplot(121)
    plt.title("Compressed")
    viewImage(compressed)
    plt.subplot(122)
    plt.title("Difference")
    viewImage(diff, cmap="gray")
    mse = MSE(original, compressed)
    rmse = RMSE(mse)
    snr = SNR(original, mse)
    psnr = PSNR(original, mse)
    print(f"MSE: {mse:.3f}\nRMSE: {rmse:.3f}")
    print(f"SNR: {snr:.3f} dB\nPSNR: {psnr:.3f} dB")
