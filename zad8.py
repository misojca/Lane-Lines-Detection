import numpy as np
import cv2 as cv
import os
from moviepy import VideoFileClip

# --- 1. UCITAVANJE PARAMETARA (Kalibracija i Perspektiva) ---
calib = np.load('result_files/calib.npz')
mtx, dist = calib['mtx'], calib['dist']

# --- 2. POMOCNE FUNKCIJE ---

def get_binary_video(img):
    """Tvoja binarizacija iz zad3 (LAB + HLS + Sobel)"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    l_channel, s_channel = hls[:,:,1], hls[:,:,2]
    
    # Zuta (LAB B-kanal) i Bela (HLS L-kanal)
    yellow_binary = np.zeros_like(b_channel)
    yellow_binary[(b_channel > 155)] = 1
    white_binary = np.zeros_like(l_channel)
    white_binary[(l_channel > 200)] = 1

    # Sobel X gradijent
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1

    combined = np.zeros_like(sx_binary)
    # Kombinovanje: prioritet boji, Sobel samo gde ima zasićenja (S > 100)
    combined[(yellow_binary == 1) | (white_binary == 1) | ((sx_binary == 1) & (s_channel > 100))] = 1
    return combined

def find_lane_pixels_video(binary_warped):
    """Sliding windows iz zad5 sa smanjenim marginom za stabilnost"""
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows, margin, minpix = 9, 70, 50 # Smanjen margin na 70 zbog ograde
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # Prozori
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)
        
        if len(good_left) > minpix: leftx_current = np.int32(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix: rightx_current = np.int32(np.mean(nonzerox[good_right]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
               nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    except:
        return None, None, None, None

# --- 3. GLAVNI PIPELINE ---

def process_frame(frame):
    # a) Undistort
    undist = cv.undistort(frame, mtx, dist, None, mtx)
    h, w = undist.shape[:2]

    # b) Binarizacija
    binary = get_binary_video(undist)

    # c) Perspektiva (zad4 tačke)
    src = np.float32([[w*0.45, h*0.62], [w*0.55, h*0.62], [w*0.85, h*0.95], [w*0.15, h*0.95]])
    dst = np.float32([[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]])
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(binary, M, (w, h), flags=cv.INTER_LINEAR)

    # d) Detekcija piksela i Fitovanje (zad5)
    lx, ly, rx, ry = find_lane_pixels_video(warped)
    
    if lx is not None and len(lx) > 0 and len(rx) > 0:
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
        
        # e) Metrika (zad6)
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        ym_per_pix, xm_per_pix = 30/720, 3.7/700
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        l_rad = ((1 + (2*left_fit_cr[0]*h*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        r_rad = ((1 + (2*right_fit_cr[0]*h*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        offset = (( (left_fitx[-1] + right_fitx[-1])/2 ) - (w/2)) * xm_per_pix

        # f) Iscrtavanje nazad na put (zad7)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv.warpPerspective(color_warp, Minv, (w, h))
        result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Ispisivanje teksta
        avg_rad = (l_rad + r_rad) / 2
        cv.putText(result, f"Curvature: {avg_rad:.1f}m", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv.putText(result, f"Offset: {offset:.2f}m", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return result
    else:
        return undist # Ako detekcija ne uspe, vrati original (da video ne stane)

# --- 4. POKRETANJE VIDEA ---
# --- 4. POKRETANJE VIDEA (Kompatibilno sa MoviePy 2.x) ---
output_video = 'result_files/final_video_output.mp4'
input_video = 'test_videos/project_video03.mp4' 

if not os.path.exists('result_files'):
    os.makedirs('result_files')

# U MoviePy 2.x fl_image se poziva preko .image_transform
with VideoFileClip(input_video) as clip:
    # Umesto clip.fl_image(process_frame)
    processed_clip = clip.image_transform(process_frame)
    processed_clip.write_videofile(output_video, audio=False)

print(f"Video je uspešno sačuvan u {output_video}")