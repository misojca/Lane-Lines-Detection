import cv2
import numpy as np

# Učitaj binarnu sliku (rezultat iz prethodnog koraka)
binary = cv2.imread('results/combined_binary_output.jpg', cv2.IMREAD_GRAYSCALE)

h, w = binary.shape

# Definiši SOURCE tačke (trapez na putu)
src = np.float32([
    [w * 0.43, h * 0.65],   # gore levo
    [w * 0.58, h * 0.65],   # gore desno
    [w * 0.90, h * 1.00],   # dole desno
    [w * 0.10, h * 1.00]    # dole levo
])

# Definiši DESTINATION tačke (pravougaonik)
dst = np.float32([
    [w * 0.25, 0],
    [w * 0.75, 0],
    [w * 0.75, h],
    [w * 0.25, h]
])

# Izračunaj matricu transformacije
M = cv2.getPerspectiveTransform(src, dst)

# Warp (bird’s-eye view)
warped = cv2.warpPerspective(binary, M, (w, h))

# Prikaz
cv2.imshow('Binary Image', binary)
cv2.imshow('Warped (Bird Eye View)', warped)

# Sačuvaj rezultat
cv2.imwrite('results/warped_binary.jpg', warped)

cv2.waitKey(0)
cv2.destroyAllWindows()
