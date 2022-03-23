from main import metrics
from PIL import Image
from os import path, mkdir


def main():
    basePath = "imagens"
    images = [
        f"{basePath}/barn_mountains.bmp",
        f"{basePath}/peppers.bmp",
        f"{basePath}/logo.bmp"
    ]

    quality = [100, 75, 50, 25, 10]

    for imagepath in images:
        name = imagepath.split("/")[-1][:-4]
        savePath= f"compressed/{name}"
        if not path.isdir(savePath):
            mkdir(savePath)
        for qf in quality:
            print(f"{name} - Quality: {qf}")
            compressed, compressionMetrics = metrics(imagepath, qf, False, False)
            Image.fromarray(compressed).save(f"{savePath}/{qf}.png")
            with open(f"{savePath}/{qf}.txt", "w") as f:
                f.write(f"MSE: {compressionMetrics['mse']}\n")
                f.write(f"RMSE: {compressionMetrics['rmse']}\n")
                f.write(f"SNR: {compressionMetrics['snr']}\n")
                f.write(f"PSNR: {compressionMetrics['psnr']}\n")


if __name__ == '__main__':
    main()