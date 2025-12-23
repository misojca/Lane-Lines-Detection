import numpy as np
import cv2 as cv
import os

# 1. Učitavanje binarne slike iz prethodnog koraka
img = cv.imread('result_files/binary_lane_image.jpg', 0)

if img is None:
    print("Greška: Binarna slika nije pronađena u result_files/")
else:
    h, w = img.shape[:2]

    # 2. Definisanje izvornih (src) i ciljnih (dst) tačaka
    # src: Tačke koje formiraju trapez na originalnoj slici (prate linije puta)
    # dst: Tačke koje formiraju pravougaonik na rezultujućoj slici
    
    # Ove tačke su pažljivo odabrane da bi na ravnom putu trake bile paralelne
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

    # 3. Izračunavanje matrice transformacije (M)
    # Koristimo cv.getPerspectiveTransform (slično kao cv.getAffineTransform u tvom fajlu)
    M = cv.getPerspectiveTransform(src, dst)
    
    # Izračunavamo i inverznu matricu (Minv) - trebaće nam kasnije da vratimo crtež na put
    Minv = cv.getPerspectiveTransform(dst, src)

    # 4. Primena transformacije (Warp)
    warped_img = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)

    # 5. Čuvanje rezultata i matrica
    if not os.path.exists('result_files'):
        os.makedirs('result_files')
    
    cv.imwrite('result_files/warped_binary_image.jpg', warped_img)
    # Čuvamo matrice u .npz formatu (kao u tvojim vezbama) jer će nam trebati u finalnom koraku
    np.savez('result_files/perspective_matrices.npz', M=M, Minv=Minv)

    # 6. Prikaz rezultata
    # Crtamo trapez na kopiji originala radi provere u Writeup-u
    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    pts = src.astype(np.int32).reshape((-1, 1, 2))
    cv.polylines(img_bgr, [pts], True, (0, 255, 0), 3)

    cv.imshow('Original with src points', img_bgr)
    cv.imshow('Warped Image (Bird-eye view)', warped_img)
    
    print("Perspective transformacija završena. Slika sačuvana.")
    cv.waitKey(0)
    cv.destroyAllWindows()