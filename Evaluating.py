import numpy as np
from FFFT_and_IFFT_Layer import *
from fourier_transform import *
import matplotlib.pyplot as plt
import cv2

print ('Testing for 1-d signals')
# Generating sin curve in range [-2p, 2pi] with 128 sample points
f = np.sin(np.linspace(-2*np.pi,2*np.pi,128))
# let us add some noise with mean 0.5 and sigma 0.75
f = f + 0.75 * np.random.rand(128) + 0.5
F = FFT_1D(f)

fig = plt.figure()
fig.add_subplot(311)
plt.plot(f)
plt.title('Original Signal')
fig.add_subplot(312)
plt.plot(np.log(np.abs(F[:64]) + 1))
plt.title('magnitude plot')
fig.add_subplot(313)
plt.plot(np.angle(F[:64]))
plt.title('Phase plot')
plt.show()

print ('\ntesting for 2-d signals/images')
x = np.matrix([[1,2,1],[2,1,2],[0,1,1]])
X, m, n = fft2(x)
print ('\nDFT is :')
print (X)
print ('\nOriginal signal is :')
print (ifft2(X, m, n))