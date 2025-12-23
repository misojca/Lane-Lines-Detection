import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1. Učitavanje warped binarne slike iz zad4.py
binary_warped = cv.imread('result_files/warped_binary_image.jpg', 0)

def find_lane_pixels(binary_warped):
    # Histogram donje polovine (bazirano na tvom histogram.py)
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Pronalazimo baze (pikove)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parametri za prozore
    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int32(binary_warped.shape[0]//nwindows)

    # Pozicije svih belih piksela
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []

    # Prolazimo kroz prozore (Sliding Windows)
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # Granice prozora
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        # Identifikuj piksele unutar prozora
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)
        
        # Ako smo nasli dovoljno piksela, pomeri centar sledeceg prozora
        if len(good_left) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right]))

    # Spajanje indeksa
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], \
           nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

# Poziv funkcije i uklapanje polinoma (Fit Polynomial)
leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

# f(y) = Ay^2 + By + C
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Cuvanje koeficijenata za sledeci zadatak
np.savez('result_files/poly_fit.npz', left_fit=left_fit, right_fit=right_fit)

# Vizuelizacija rezultata
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.show()