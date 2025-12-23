# binary_threshold.py
import cv2
import numpy as np

# Učitaj distorziju-korigovanu sliku (ili neku test sliku iz foldera)
img = cv2.imread('test_images/test1.jpg')  # promeni putanju po potrebi

calib = np.load('calib.npz')
mtx = calib['mtx']
dist = calib['dist']
h, w = img.shape[:2]
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
img = cv2.undistort(img, mtx, dist, None, newCameraMtx)

# Konvertuj u HLS (za S-kanal)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
s_channel = hls[:, :, 2]

# Konvertuj u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Primeni Sobel X na grayscale
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

# Threshold za Sobel X
thresh_min = 20
thresh_max = 100
binary_sobel = np.zeros_like(scaled_sobel)
binary_sobel[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold za S-kanal
s_thresh_min = 170
s_thresh_max = 255
binary_s = np.zeros_like(s_channel)
binary_s[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Kombinuj obe binarne slike
combined_binary = np.zeros_like(binary_s)
combined_binary[(binary_sobel == 1) | (binary_s == 1)] = 1

# Prikaz rezultata
#cv2.imshow('Original', img)
#cv2.imshow('Sobel X Binary', binary_sobel * 255)
#cv2.imshow('S-channel Binary', binary_s * 255)
cv2.imshow('Combined Binary', combined_binary * 255)

output_image = (combined_binary * 255).astype(np.uint8)

# Sačuvaj sliku u folder
cv2.imwrite('results/combined_binary_output.jpg', output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
