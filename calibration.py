import numpy as np
import cv2
import glob
import os

def get_calibration_params():

    calib_path = 'result_files/calib.npz'

    if os.path.exists(calib_path):
        print("Loading calibration parameters")
        data = np.load(calib_path)
        return data['mtx'], data['dist']
    
    rows = 6
    cols = 9

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Create the arrays to store the object points and the image points
    objectPointsArray = []
    imgPointsArray = []

    # Loop over the image files (koristimo putanju tvojih slika za kalibraciju)
    images = glob.glob('camera_cal/calibration*.jpg')

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # Make sure the chess board pattern was found in the image
        if ret:
            # Refine the corner position
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Add the object points and the image points to the arrays
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            # Draw the corners on the image
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        
        # Display the image
       # cv2.imshow('chess board', img) # Promenjeno na 'img' da vidiš uglove u boji
      #  cv2.waitKey(20)

    cv2.destroyAllWindows()

    # Calibrate the camera and save the results
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    
    # Kreiraj folder ako ne postoji i sačuvaj
    if not os.path.exists('result_files'):
        os.makedirs('result_files')
    np.savez(calib_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Print the camera calibration error
    error = 0
    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    print("Total error: ", error / len(objectPointsArray))
    
    return mtx, dist