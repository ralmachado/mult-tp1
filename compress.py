from main import metrics
from PIL import Image


def main():
    basePath = "imagens"
    savePath= "compressed"
    images = [
        f"{basePath}/barn_mountains.bmp",
        f"{basePath}/peppers.bmp",
        f"{basePath}/logo.bmp"
    ]

    quality = [100, 75, 50, 25, 10]

    for path in images:
        name = path.split("/")[-1][:-4]
        for qf in quality:
            compressed, compressionMetrics = metrics(path, qf, False, False)
            Image.fromarray(compressed).save(f"{savePath}/{name}_{qf}.png")
            with open(f"{savePath}/{name}_{qf}.txt", "w") as f:
                f.write(f"MSE: {compressionMetrics['mse']}\n")
                f.write(f"RMSE: {compressionMetrics['rmse']}\n")
                f.write(f"SNR: {compressionMetrics['snr']}\n")
                f.write(f"PSNR: {compressionMetrics['psnr']}\n")


if __name__ == '__main__':
    main()