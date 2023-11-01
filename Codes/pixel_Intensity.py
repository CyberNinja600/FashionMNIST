import numpy as np
import matplotlib.pyplot as plt

def pixel_intensity_analysis(images):
    images_flat = images.reshape(images.shape[0], -1)

    # Calculate the mean and standard deviation
    mean_intensity = np.mean(images_flat, axis=1)
    std_intensity = np.std(images_flat, axis=1)

    # Plot histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(mean_intensity, bins=50, color='b', alpha=0.7)
    plt.title('Mean Pixel Intensity Distribution')
    plt.xlabel('Mean Pixel Intensity')


    plt.subplot(1, 2, 2)
    plt.hist(std_intensity, bins=50, color='g', alpha=0.7)
    plt.title('Standard Deviation Pixel Intensity Distribution')
    plt.xlabel('Standard Deviation Pixel Intensity')


    plt.tight_layout()
    plt.show()


