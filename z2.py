import cv2
import numpy as np

# Učitaj kalibracione koeficijente
calib = np.load('calib.npz')
mtx = calib['mtx']
dist = calib['dist']

# Učitaj test sliku
img = cv2.imread('camera_cal/calibration1.jpg')  # promeni putanju ako treba
h, w = img.shape[:2]

# Dobij optimalnu novu kameru matricu
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Ukloni distorziju
undistorted = cv2.undistort(img, mtx, dist, None, newCameraMtx)

# Prikaz originalne i ispravljene slike
cv2.imshow('Original vs Undistorted', np.hstack((img, undistorted)))
cv2.waitKey(0)
cv2.destroyAllWindows()
