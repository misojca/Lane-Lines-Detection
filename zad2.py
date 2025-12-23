import numpy as np
import cv2
import glob

# Load calibrated camera parameters
calibratio = np.load('result_files/calib.npz')
mtx = calibratio['mtx']
dist = calibratio['dist']
rvecs = calibratio['rvecs']
tvecs = calibratio['tvecs']

# Load one of the test images
img = cv2.imread('test_images/challange00101.jpg')
h, w = img.shape[:2]

# Obtain the new camera matrix and undistort the image
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
# cv2.imwrite('result_files/distortion_corrected_img.jpg', np.hstack((img, undistortedImg)))
cv2.imwrite('result_files/distortion_corrected_img.jpg', undistortedImg)
# Display the final result
cv2.imshow('Result image', np.hstack((img, undistortedImg)))
cv2.waitKey(0)
cv2.destroyAllWindows()
