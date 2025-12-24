import numpy as np
import cv2 as cv
import os

def get_binary_image(img):
    # LAB prostor - B kanal je najbolji za zutu boju
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    
    # L kanal iz HLS za osvetljenje (bele linije)
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # --- POSEBAN TRETMAN ZA ZUTU ---
    # Zuta linija je obicno > 155 u B kanalu LAB-a
    yellow_binary = np.zeros_like(b_channel)
    yellow_binary[(b_channel > 155)] = 1

    # --- POSEBAN TRETMAN ZA BELU ---
    # Bela linija u daljini je svetla (L) i ima neki gradijent
    white_binary = np.zeros_like(l_channel)
    white_binary[(l_channel > 200)] = 1

    # Gradijent (Sobel) - smanjujemo mu vaznost da ne hvata ogradu
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 100)] = 1

    # KOMBINOVANJE: Prioritet dajemo boji (yellow/white)
    combined = np.zeros_like(sx_binary)
    # Sobel koristimo samo tamo gde je slika bar malo "zasicena" (S kanal) 
    # To filtrira sive senke jer one imaju nultu saturaciju
    combined[(yellow_binary == 1) | (white_binary == 1) | ((sx_binary == 1) & (s_channel > 100))] = 1
    
    return combined * 255

# --- Glavni deo koda ---
img = cv.imread('result_files/distortion_corrected_img.jpg')

if img is not None:
    binary_img = get_binary_image(img)
    
    if not os.path.exists('result_files'):
        os.makedirs('result_files')
        
    cv.imwrite('result_files/binary_lane_image.jpg', binary_img)
    
    # Vizuelna provera
    cv.imshow('Binary bez senki (HLS + LAB)', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()