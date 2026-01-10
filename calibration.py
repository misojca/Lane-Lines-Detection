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

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    objectPointsArray = []
    imgPointsArray = []

    # Kalibracija se primenjuje na slike iz direktorijuma camera_cal
    images = glob.glob('camera_cal/calibration*.jpg')

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Nalazenje uglova na tabli
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)

            # Crtanje uglova na slici
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
        

    cv2.destroyAllWindows()

    # Cuvanje rezultata kalibracije
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    
    # Kreiranje foldera ako ne postoji
    if not os.path.exists('result_files'):
        os.makedirs('result_files')
    np.savez(calib_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    # Ispis greske pri kalibraciji
    error = 0
    for i in range(len(objectPointsArray)):
        imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

    print("Total error: ", error / len(objectPointsArray))
    
    return mtx, dist