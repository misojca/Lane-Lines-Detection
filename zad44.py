import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1. Učitavanje Bird's-eye slike iz zad4.py
binary_warped = cv.imread('result_files/warped_binary_image.jpg', 0)

def find_lane_pixels(binary_warped):
    # Uzimamo histogram donje polovine slike (kao u tvom histogram.py)
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Pronalazimo baze leve i desne trake (najveći pikovi na histogramu)
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parametri za sliding windows
    nwindows = 9 # Broj prozora odozdo do gore
    margin = 100 # Širina prozora
    minpix = 50  # Minimum piksela da bi se prozor pomerio

    # Visina jednog prozora
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    
    # Pozicije svih belih piksela na slici
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Trenutne pozicije koje ćemo ažurirati za svaki prozor
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Prolazimo kroz prozore odozdo nagore
    for window in range(nwindows):
        # Definišemo granice prozora (y i x)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identifikujemo bele piksele unutar prozora
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Ako smo našli dovoljno piksela, pomeramo centar sledećeg prozora (dinamičko praćenje)
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Spajamo indekse u jedan niz
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Izdvajamo pozicije piksela leve i desne trake
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

# 2. Uklapanje polinoma (Fit Polynomial)
leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

# Uklapamo kvadratnu funkciju: f(y) = Ay^2 + By + C
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generisanje x vrednosti za crtanje
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# 3. Prikaz rezultata
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()

# Čuvanje koeficijenata za kasnije
np.savez('result_files/poly_fit.npz', left_fit=left_fit, right_fit=right_fit)