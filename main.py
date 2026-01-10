import numpy as np
import cv2 as cv
import os
import glob
from calibration import get_calibration_params

def get_binary_video(img):
    # Konverzija u LAB prostor
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel = lab[:,:,0] # Lightness (za bele linije)
    b_channel = lab[:,:,2] # Blue-Yellow (za zute linije)
    
    # Detekcija zute boje (B kanal u LAB je koristan za ovo)
    yellow_binary = np.zeros_like(b_channel)
    yellow_binary[(b_channel > 155)] = 1
    
    # Detekcija bele boje (L kanal)
    white_binary = np.zeros_like(l_channel)
    white_binary[(l_channel > 210)] = 1 # Visok prag za preciznost
    
    # Sobel gradijent po x-osi (ivice)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1
    
    # Kombinovanje: zuta , bela, Sobel
    combined = np.zeros_like(sx_binary)
    combined[(yellow_binary == 1) | (white_binary == 1) | ((sx_binary == 1) & (l_channel > 150))] = 1
    
    return combined

def find_lane_pixels_video(binary_warped, draw_lanes=False):
    # Racunanje histograma donje polovine slike kako bismo identifikovali pocetne pozicije leve i desne trake
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
    # Iteracija kroz prozore od dna ka vrhu slike radi pracenja zakrivljenosti traka
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
        # Spajanje svih pronadjenih indeksa piksela i izdvajanje njihovih koordinata za fitovanje polinoma
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        out_img = None
        if draw_lanes:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255] # Crvena
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0] # Plava

        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds], out_img
    except: 
        return None, None, None, None, None

def process_frame(frame, mtx, dist):
    h, w = frame.shape[:2]
    undist = cv.undistort(frame, mtx, dist, None, mtx)
    binary = get_binary_video(undist)
    
    # Definisanje izvornih i odredisnih tacaka za transformaciju slike u pticju perspektivu
    src = np.float32([[w*0.45, h*0.62], [w*0.55, h*0.62], [w*0.85, h*0.95], [w*0.15, h*0.95]])
    dst = np.float32([[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]])
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(binary, M, (w, h), flags=cv.INTER_LINEAR)

    lx, ly, rx, ry, _ = find_lane_pixels_video(warped)
    
    if lx is not None and len(lx) > 0 and len(rx) > 0:
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Konverzija piksela u metre i izracunavanje radijusa zakrivljenosti i pozicije vozila (offset)
        ym_per_pix, xm_per_pix = 30/720, 3.7/700
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        
        l_rad = ((1 + (2*left_fit_cr[0]*h*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        r_rad = ((1 + (2*right_fit_cr[0]*h*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        offset = (((left_fitx[-1] + right_fitx[-1])/2) - (w/2)) * xm_per_pix

        # Iscrtavanje detektovane povrsine trake i vracanje maske u originalnu perspektivu kamere
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
        return result
    return undist

def process_video(input_path, output_path, mtx, dist):

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = process_frame(frame, mtx, dist)
        out.write(processed)
        
        # Prikazivanje prozora tokom obrade 
        cv.imshow('Lane Detection Video', processed)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print(f"Obrada zavrsena! Video sacuvan kao: {output_path}")

def main():
    if not os.path.exists('result_files'):
        os.makedirs('result_files')

    mtx, dist = get_calibration_params()

    cal_img = cv.imread('camera_cal/calibration1.jpg')
    if cal_img is not None:
        cv.imwrite('result_files/calibration_original.jpg', cal_img)
        cal_undist = cv.undistort(cal_img, mtx, dist, None, mtx)
        cv.imwrite('result_files/calibration_undistorted.jpg', cal_undist)

    # Ulazna slika koriscena za writeup_template.md
    test_img = cv.imread('test_images/test3.jpg')
    if test_img is not None:
        h, w = test_img.shape[:2]
        
        # 1. Primena kalibracije na ulaznu sliku
        demo_undist = cv.undistort(test_img, mtx, dist, None, mtx)
        cv.imwrite('result_files/road_undistorted.jpg', demo_undist)
        cv.imshow('Undistort', demo_undist)
        cv.waitKey(0)
        
        # 2. Binarna slika
        demo_binary = get_binary_video(demo_undist)
        cv.imwrite('result_files/binary_thresholded.jpg', demo_binary * 255)
        cv.imshow('Binary', demo_binary * 255)
        cv.waitKey(0)
        
        # 3. Perspektivna transformacija
        src = np.float32([[w*0.45, h*0.62], [w*0.55, h*0.62], [w*0.85, h*0.95], [w*0.15, h*0.95]])
        dst = np.float32([[w*0.2, 0], [w*0.8, 0], [w*0.8, h], [w*0.2, h]])
        M = cv.getPerspectiveTransform(src, dst)
        demo_warped = cv.warpPerspective(demo_binary, M, (w, h))
        cv.imwrite('result_files/perspective_warped.jpg', demo_warped * 255)
        cv.imshow('Perspective Transform', demo_warped * 255)
        cv.waitKey(0)
        
        # 4. Detekcija trake na putu
        lx, ly, rx, ry, demo_lanes_img = find_lane_pixels_video(demo_warped, draw_lanes=True)
        if demo_lanes_img is not None:
            left_fit = np.polyfit(ly, lx, 2)
            right_fit = np.polyfit(ry, rx, 2)
            ploty = np.linspace(0, h-1, h)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            for i in range(len(ploty)-1):
                cv.line(demo_lanes_img, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 3)
                cv.line(demo_lanes_img, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (0, 255, 255), 3)

            cv.imwrite('result_files/lane_pixels_fitted.jpg', demo_lanes_img)
            cv.imshow('Lane line detection', demo_lanes_img)
            cv.waitKey(0)
        
        # 5. Finalna slika
        demo_final = process_frame(test_img, mtx, dist)
        cv.imwrite('result_files/final_result_image.jpg', demo_final)
        cv.imshow('Final image', demo_final)

        cv.waitKey(0)
        cv.destroyAllWindows()

    # Primena na video snimku
    videos_to_process = [
        ('test_videos/project_video01.mp4', 'result_files/output_video01.avi'),
        ('test_videos/project_video02.mp4', 'result_files/output_video02.avi')
    ]

    for input_v, output_v in videos_to_process:
        process_video(input_v, output_v, mtx, dist)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()