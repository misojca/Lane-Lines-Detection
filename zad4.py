import numpy as np
import cv2 as cv
import os

# Definicija funkcije iz primera
def warper(img, src, dst):
    # Uzimanje dimenzija slike
    img_size = (img.shape[1], img.shape[0])
    # Računanje matrice transformacije
    M = cv.getPerspectiveTransform(src, dst)
    # Primena transformacije (koristimo INTER_LINEAR za glatkiji prelaz ili INTER_NEAREST po primeru)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)
    
    return warped

# 1. Učitavanje binarne slike
img = cv.imread('result_files/binary_lane_image.jpg', 0)

if img is None:
    print("Greška: Binarna slika nije pronađena!")
else:
    h, w = img.shape[:2]

    # 2. Definisanje tačaka (izvornih i ciljnih)
    src = np.float32([
        [w * 0.45, h * 0.62], # Gore levo
        [w * 0.55, h * 0.62], # Gore desno
        [w * 0.85, h * 0.95], # Dole desno
        [w * 0.15, h * 0.95]  # Dole levo
    ])

    dst = np.float32([
        [w * 0.2, 0],   # Gore levo
        [w * 0.8, 0],   # Gore desno
        [w * 0.8, h],   # Dole desno
        [w * 0.2, h]    # Dole levo
    ])

    # 3. Pozivanje warper funkcije
    warped_img = warper(img, src, dst)

    # 4. Izračunavanje Minv (treba nam za zad7.py da vratimo sliku nazad)
    Minv = cv.getPerspectiveTransform(dst, src)
    # Čuvamo i matricu M (ponovo izračunatu) da bismo imali kompletan npz fajl
    M = cv.getPerspectiveTransform(src, dst)

    # 5. Čuvanje rezultata
    if not os.path.exists('result_files'):
        os.makedirs('result_files')
    
    cv.imwrite('result_files/warped_binary_image.jpg', warped_img)
    np.savez('result_files/perspective_matrices.npz', M=M, Minv=Minv)

    # 6. Prikaz (opciono)
    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    pts = src.astype(np.int32).reshape((-1, 1, 2))
    cv.polylines(img_bgr, [pts], True, (0, 255, 0), 3)

    cv.imshow('Original with src points', img_bgr)
    cv.imshow('Warped Image (Function used)', warped_img)
    
    print("Perspective transformacija pomoću funkcije warper je uspešna.")
    cv.waitKey(0)
    cv.destroyAllWindows()