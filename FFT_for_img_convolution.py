import cv2
import numpy as np
import matplotlib.pyplot as plt
from filters import *
from FFFT_and_IFFT_Layer import *

import os

base_folder = "result"
sub_folder = "fft_result"
output_filename = "img1.jpg"
output_path = os.path.join(base_folder, sub_folder)
os.makedirs(output_path, exist_ok=True)

# Full path for saving the image
output_filepath = os.path.join(output_path, output_filename)

# Load grayscale image
img_c1 = cv2.imread("input_images/img1.jpeg", 0)

# Apply FFT using custom function
img_c2, m, n = fft2(img_c1)  # Compute FFT
img_c3 = fftshift(img_c2)  # Center the spectrum
img_c4 = ifftshift(img_c3)  # Decentralize the spectrum
img_c5 = ifft2(img_c4, m, n)  # Apply custom IFFT

# Apply Low-Pass Filter (Custom)
LPF_custom = idealFilterLP(50, img_c3.shape)
LowPassCenter_custom = img_c3 * LPF_custom
LowPassDecentralized_custom = ifftshift(LowPassCenter_custom)
inverse_LowPass_custom = ifft2(LowPassDecentralized_custom, m, n)

# Apply High-Pass Filter (Custom)
HPF_custom = idealFilterHP(50, img_c3.shape)
HighPassCenter_custom = img_c3 * HPF_custom
HighPassDecentralized_custom = ifftshift(HighPassCenter_custom)
inverse_HighPass_custom = ifft2(HighPassDecentralized_custom, m, n)


# Apply Gaussian Low-Pass Filter (Predefined)
LPF_predefined = gaussianLP(50, img_c3.shape)  # Ensures filter matches FFT size
LowPassCenter_predefined = img_c3 * LPF_predefined
LowPass_predefined = ifftshift(LowPassCenter_predefined)  # Use custom inverse shift
inverse_LowPass_predefined = ifft2(LowPass_predefined, m, n)  # Use predefined IFFT

# Apply Gaussian High-Pass Filter (Predefined)
HPF_predefined = gaussianHP(50, img_c3.shape)  # Ensures filter matches FFT size
HighPassCenter_predefined = img_c3 * HPF_predefined
HighPass_predefined = ifftshift(HighPassCenter_predefined)
inverse_HighPass_predefined = ifft2(HighPass_predefined, m, n)

# Plot all results in a single figure
plt.figure(figsize=(14, 8))

# First row (Original + Spectral Transform)
plt.subplot(3, 4, 1), plt.imshow(img_c1, "gray"), plt.title("Original Image")
plt.subplot(3, 4, 2), plt.imshow(np.log(1 + np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(3, 4, 3), plt.imshow(np.log(1 + np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(3, 4, 4), plt.imshow(np.log(1 + np.abs(img_c4)), "gray"), plt.title("Decentralized Spectrum")

# Second row (Custom FFT + IFFT Processed Images)
plt.subplot(3, 4, 5), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")
plt.subplot(3, 4, 6), plt.imshow(np.abs(inverse_LowPass_custom), "gray"), plt.title("Low-Pass Filter (Custom)")
plt.subplot(3, 4, 7), plt.imshow(np.abs(inverse_HighPass_custom), "gray"), plt.title("High-Pass Filter (Custom)")

# Third row (Predefined FFT + IFFT Processed Images)
plt.subplot(3, 4, 8), plt.imshow(np.abs(inverse_LowPass_predefined), "gray"), plt.title("Gaussian LPF")
plt.subplot(3, 4, 9), plt.imshow(np.abs(inverse_HighPass_predefined), "gray"), plt.title("Gaussian HPF")
diff = np.abs(img_c3 - img_c4)
plt.subplot(3, 4, 10), plt.imshow(np.log(1 + diff), "gray"), plt.title("Difference Between Centered & Decentralized")

# Adjust layout
plt.tight_layout()
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
plt.show()
