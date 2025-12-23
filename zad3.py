import numpy as np
import cv2 as cv
import os

def get_binary_image(img):
    # 1. Konverzija u HLS prostor (iz tvog color_spaces.py)
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    s_channel = hls[:,:,2] # Saturation kanal
    l_channel = hls[:,:,1] # Lightness kanal

    # 2. Sobel X gradijent (iz tvog sobel.py)
    # Radimo grayscale pa Sobel da detektujemo vertikalne ivice
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) 
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Prag za Sobel X (detektuje linije na osnovu kontrasta)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # 3. Prag za S kanal (detektuje linije na osnovu boje/zasićenosti)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    # 4. Kombinovanje (Logičko "ILI")
    # Rezultat je 1 ako je ispunjen uslov gradijenta ILI uslov boje
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary * 255 # Vraćamo u opseg 0-255 za prikaz

# --- Glavni deo koda ---
# Učitavamo ispravljenu sliku iz prethodnog zadatka
img = cv.imread('result_files/distortion_corrected_img.jpg')

if img is not None:
    binary_img = get_binary_image(img)
    
    # Čuvanje rezultata
    cv.imwrite('result_files/binary_lane_image.jpg', binary_img)
    
    # Prikaz (prikazujemo S kanal i Sobel uporedo da vidiš razliku)
    cv.imshow('Original', img)
    cv.imshow('Binary Lane Detection', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()