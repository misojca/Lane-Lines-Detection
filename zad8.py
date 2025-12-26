import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt

# --- 1. FUNKCIJA ZA KALIBRACIJU ---
def get_calibration_params():
    calib_file = 'result_files/calib.npz'
    if os.path.exists(calib_file):
        calib = np.load(calib_file)
        return calib['mtx'], calib['dist']
    
    print("Clibration file not found. Starting calibration")
    rows, cols = 5, 9
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objectPointsArray, imgPointsArray = [], []
    
    images = glob.glob('camera_cal/calibration*.jpg')
    for path in images:
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objectPointsArray.append(objectPoints)
            imgPointsArray.append(corners)
            cv.drawChessboardCorners(img, (rows, cols), corners, ret)
            cv.imshow('Calibration', img)
            cv.waitKey(50)
    
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
    if not os.path.exists('result_files'): os.makedirs('result_files')
    np.savez(calib_file, mtx=mtx, dist=dist)
    return mtx, dist

# --- 2. POMOCNE FUNKCIJE ---

def get_binary_video(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    l_channel, s_channel = hls[:,:,1], hls[:,:,2]
    yellow_binary = np.zeros_like(b_channel)
    yellow_binary[(b_channel > 155)] = 1
    white_binary = np.zeros_like(l_channel)
    white_binary[(l_channel > 200)] = 1
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1
    combined = np.zeros_like(sx_binary)
    combined[(yellow_binary == 1) | (white_binary == 1) | ((sx_binary == 1) & (s_channel > 100))] = 1
    return combined

def find_lane_pixels_video(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows, margin, minpix = 9, 70, 50 
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)
        if len(good_left) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right]))
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    except: return None, None, None, None

def process_frame(frame, mtx, dist, return_all=False):
    h, w = frame.shape[:2]
    # Undistorted image
    undist = cv.undistort(frame, mtx, dist, None, mtx)
    # Binary Image
    binary = get_binary_video(undist)
    # Perspective transform
    src = np.float32([[w*0.45, h*0.62], [w*0.55, h*0.62], [w*0.85, h*0.95], [w*0.15, h*0.95]])
    dst = np.float32([[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]])
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(binary, M, (w, h), flags=cv.INTER_LINEAR)

    lx, ly, rx, ry = find_lane_pixels_video(warped)
    
    if lx is not None and len(lx) > 0 and len(rx) > 0:
        # Zad5: Fitting
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Zad6: Curvature
        ym_per_pix, xm_per_pix = 30/720, 3.7/700
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        l_rad = ((1 + (2*left_fit_cr[0]*h*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        r_rad = ((1 + (2*right_fit_cr[0]*h*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        offset = (((left_fitx[-1] + right_fitx[-1])/2) - (w/2)) * xm_per_pix

        # Zad7: Draw back
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv.warpPerspective(color_warp, Minv, (w, h))
        result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        cv.putText(result, f"Radius: {(l_rad+r_rad)/2:.1f}m", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv.putText(result, f"Offset: {offset:.2f}m", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if return_all: return undist, binary, warped, result
        return result
    return undist

# --- 3. GLAVNI PROGRAM ---

# KORAK 1: Kalibracija
mtx, dist = get_calibration_params()

# KORAK 2: Prikaz koraka na ulaznoj slici
print("\nPrikazujem rezultate za 'test_images/test2.jpg'...")
test_img = cv.imread('test_images/test2.jpg')
if test_img is not None:
    undist, binary, warped, final_img = process_frame(test_img, mtx, dist, return_all=True)
    
    # Prikaz koraka 
    cv.imshow('Zad2: Undistorted Image', undist)
    cv.waitKey(0)
    cv.imshow('Zad3: Thresholded Binary', binary * 255)
    cv.waitKey(0)
    cv.imshow('Zad4: Birds-eye View', warped * 255)
    cv.waitKey(0)
    cv.imshow('Zad5-7: Final Result on Image', final_img)
    print("Pritisnite bilo koji taster na slici da krenete na video...")
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Image not founf'!")

# KORAK 3: Video procesiranje
input_path = 'test_videos/project_video01.mp4'
output_path = 'result_files/final_video_opencv.avi'
cap = cv.VideoCapture(input_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

print("\nStarting video processing")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    processed = process_frame(frame, mtx, dist)
    out.write(processed)
    cv.imshow('Final Video Pipeline', processed)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv.destroyAllWindows()
print(f" --- Video saved in: {output_path} ---")