import cv2 as cv
import numpy as np
import os

# ===== UČITAVANJE PODATAKA =====
calib = np.load('result_files/calib.npz')
mtx, dist = calib['mtx'], calib['dist']

persp = np.load('result_files/perspective_matrices.npz')
M, Minv = persp['M'], persp['Minv']

# ===== POMOĆNE FUNKCIJE =====

def get_binary_image(img):
    # Pretvaramo u HLS jer je S kanal najstabilniji za bele linije
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    s = hls[:, :, 2]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Kombinujemo gradijent i boju
    sxbinary = ((scaled >= 20) & (scaled <= 100))
    sbinary = ((s >= 170) & (s <= 255))

    binary_output = np.zeros_like(s)
    binary_output[(sxbinary | sbinary)] = 255
    return binary_output

def find_lane_pixels(binary_warped):
    # Histogram donje polovine slike
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Postavke kliznih prozora (Sliding Windows)
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        # Identifikacija piksela unutar prozora
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return lefty, leftx, righty, rightx

def calculate_metrics(img_shape, left_fit, right_fit):
    # Definisanje konverzije piksel -> metar
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/700
    y_eval = img_shape[0] - 1
    
    # Radijus krivine u metrima
    left_fit_cr = np.polyfit(np.linspace(0, img_shape[0]-1, img_shape[0])*ym_per_pix, 
                             (left_fit[0]*np.linspace(0, img_shape[0]-1, img_shape[0])**2 + 
                              left_fit[1]*np.linspace(0, img_shape[0]-1, img_shape[0]) + left_fit[2])*xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.linspace(0, img_shape[0]-1, img_shape[0])*ym_per_pix, 
                              (right_fit[0]*np.linspace(0, img_shape[0]-1, img_shape[0])**2 + 
                               right_fit[1]*np.linspace(0, img_shape[0]-1, img_shape[0]) + right_fit[2])*xm_per_pix, 2)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    radius = (left_curverad + right_curverad) / 2

    # Pozicija vozila u odnosu na centar
    car_pos = img_shape[1] / 2
    lane_center = ( (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]) + 
                    (right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]) ) / 2
    offset = (lane_center - car_pos) * xm_per_pix
    
    return radius, offset

def project_back_to_road_video(undist, left_fit, right_fit, Minv, radius, offset):
    h, w = undist.shape[:2]
    ploty = np.linspace(0, h-1, h)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros((h, w), dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.vstack([left_fitx, ploty]).T
    pts_right = np.vstack([right_fitx, ploty]).T
    pts = np.vstack([pts_left, pts_right[::-1]])
    pts = np.int32([pts])

    # Popunjavanje zelene površine
    cv.fillPoly(color_warp, pts, (0, 255, 0))
    # Iscrtavanje žutih linija sa strana (baš kao na tvojoj slici)
    cv.polylines(color_warp, [np.int32(pts_left)], False, (0, 255, 255), 15)
    cv.polylines(color_warp, [np.int32(pts_right)], False, (0, 255, 255), 15)

    newwarp = cv.warpPerspective(color_warp, Minv, (w, h))
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Ispis teksta
    cv.putText(result, f"Radius of Curvature: {radius:.2f}m", (50, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    direction = "left" if offset < 0 else "right"
    cv.putText(result, f"Vehicle is {abs(offset):.2f}m {direction} of center", (50, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result

# ===== VIDEO PIPELINE =====
video_path = 'test_videos/project_video01.mp4'
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Greška: Ne mogu da otvorim video!")
    exit()

h, w = int(cap.get(4)), int(cap.get(3))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('result_files/final_video.avi', fourcc, 25.0, (w, h))

print("Obrada videa je započeta...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Ispravka distorzije
    undist = cv.undistort(frame, mtx, dist)
    
    # 2. Binarna slika
    binary = get_binary_image(undist)
    
    # 3. Bird's eye view
    warped = cv.warpPerspective(binary, M, (w, h))

    try:
        # 4. Pronalaženje piksela i fitovanje polinoma
        ly, lx, ry, rx = find_lane_pixels(warped)
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
        
        # 5. Metrika
        radius, offset = calculate_metrics((h, w), left_fit, right_fit)
        
        # 6. Projekcija nazad na put pomoću tvoje funkcije
        result = project_back_to_road_video(undist, left_fit, right_fit, Minv, radius, offset)
    except Exception as e:
        # Fallback ako detekcija ne uspe u nekom frejmu
        result = undist 

    out.write(result)

cap.release()
out.release()
print("✅ Video uspešno sačuvan: result_files/final_video.avi")