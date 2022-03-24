from main import *
import matplotlib.pyplot as plt

def verboseMetrics(original: np.ndarray, qf: int = 75, ratio: tuple = (4,2,0)):
    y, cb, cr, shape, yOriginal = encoder(original, ratio, qf=qf, verbose=True)
    compressed, yReconstructed = decoder((y,cb,cr), shape, qf=qf, verbose=True)
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


def viewComparison(image: np.ndarray, other: np.ndarray) -> None:
    plt.subplot(121)
    plt.title("Original")
    plt.axis("off")
    plt.imshow(image)
    plt.subplot(122)
    plt.title("Compressed")
    plt.axis("off")
    plt.imshow(other)
