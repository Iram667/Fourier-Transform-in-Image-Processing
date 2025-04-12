import cv2
import numpy as np
import matplotlib.pyplot as plt
from FFFT_and_IFFT_Layer import *  # Custom FFT functions
from filters import gaussianLP  # Import Gaussian Low-Pass Filter
import os

base_folder = "result"
sub_folder = "ft_noise_removal"
output_filename = "img3_result.jpg"
output_path = os.path.join(base_folder, sub_folder)
os.makedirs(output_path, exist_ok=True)
output_filepath = os.path.join(output_path, output_filename)

img = cv2.imread("input_images/image 3.PNG", 0)

# Add artificial noise (simulating high-frequency noise)
noise = np.random.normal(0, 50, img.shape)  # Gaussian noise
noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255)  # Ensure values remain valid [0,255]

# Apply FFT to noisy image
noisy_fft, m, n = fft2(noisy_img)
centered_fft = fftshift(noisy_fft)  # Shift FFT for visualization

# Apply Low-Pass Filter (LPF) to remove noise
D0 = 50  # Cutoff frequency (tune this to adjust denoising strength)
LPF = gaussianLP(D0, centered_fft.shape)
filtered_fft = centered_fft * LPF  # Apply LPF in frequency domain

# Apply Inverse FFT to get back denoised image
decentralized_fft = ifftshift(filtered_fft)  # Undo FFT shift
denoised_img = ifft2(decentralized_fft, m, n)  # Convert back to spatial domain

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title("Original Image")
plt.subplot(1, 4, 2), plt.imshow(noisy_img, cmap='gray'), plt.title("Noisy Image")
plt.subplot(1, 4, 3), plt.imshow(np.log(1 + np.abs(centered_fft)), cmap='gray'), plt.title("Spectrum (Noisy Image)")
plt.subplot(1, 4, 4), plt.imshow(np.abs(denoised_img), cmap='gray'), plt.title("Denoised Image")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
plt.show()
