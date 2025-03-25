import cv2
import numpy as np
import matplotlib.pyplot as plt
from FFFT_and_IFFT_Layer import *  # Custom FFT functions
import os


base_folder = "result"
sub_folder = "ft_compression"
output_filename = "img3_result.jpg"
output_path = os.path.join(base_folder, sub_folder)
os.makedirs(output_path, exist_ok=True)  # Ensure both folders exist
output_filepath = os.path.join(output_path, output_filename)

img = cv2.imread("input_images/img3.jpg", 0)

# Apply FFT
fft_image, m, n = fft2(img)
fft_shifted = fftshift(fft_image)  # Center the low frequencies

# Create a Mask to Keep Only Low Frequencies (Compression)
compression_ratio = 0.2
rows, cols = fft_shifted.shape
crow, ccol = rows // 2, cols // 2  # Find the center

# Generate mask (keep only a small portion of the frequency domain)
mask = np.zeros((rows, cols), np.uint8)
mask[crow-int(compression_ratio*crow):crow+int(compression_ratio*crow),
     ccol-int(compression_ratio*ccol):ccol+int(compression_ratio*ccol)] = 1

# Apply the mask (Remove high-frequency components)
compressed_fft = fft_shifted * mask

# Apply Inverse FFT
ifft_shifted = ifftshift(compressed_fft)  # Move frequencies back
compressed_img = ifft2(ifft_shifted, m, n)  # Convert back to spatial domain

# Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
plt.subplot(1, 3, 2), plt.imshow(np.log(1 + np.abs(fft_shifted)), cmap='gray'), plt.title("Full Spectrum")
plt.subplot(1, 3, 3), plt.imshow(np.abs(compressed_img), cmap='gray'), plt.title("Compressed Image")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
plt.show()
