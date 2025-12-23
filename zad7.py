import numpy as np
import cv2 as cv

# 1. Učitavanje potrebnih podataka iz prethodnih koraka
# Originalna ispravljena slika
undist = cv.imread('result_files/distortion_corrected_img.jpg')
# Binarna warped slika (samo za dimenzije)
warped = cv.imread('result_files/warped_binary_image.jpg', 0)
# Matrice transformacije
matrices = np.load('result_files/perspective_matrices.npz')
Minv = matrices['Minv']
# Koeficijenti polinoma
poly_data = np.load('result_files/poly_fit.npz')
left_fit = poly_data['left_fit']
right_fit = poly_data['right_fit']
# Metrički podaci
metrics = np.load('result_files/metrics.npz')
radius = metrics['radius']
offset = metrics['offset']

def project_back_to_road(undist, warped, left_fit, right_fit, Minv):
    # Generisanje y vrednosti
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    
    # Računanje x vrednosti polinoma
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Kreiranje prazne slike za crtanje trake (iste dimenzije kao warped)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Priprema tačaka za ispunu (zelena boja između traka)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Crtanje popunjene zelene površine i žutih ivica
    cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv.polylines(color_warp, np.int_([pts_left]), False, (0, 255, 255), 15)
    cv.polylines(color_warp, np.int_([pts_right]), False, (0, 255, 255), 15)

    # Vraćanje u originalnu perspektivu pomoću Minv
    newwarp = cv.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    
    # Kombinovanje rezultata sa originalnom ispravljenom slikom
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Dodavanje teksta (Krivina i Offset) na finalnu sliku
    cv.putText(result, f"Radius of Curvature: {radius:.2f}m", (50, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    direction = "left" if offset < 0 else "right"
    cv.putText(result, f"Vehicle is {abs(offset):.2f}m {direction} of center", (50, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

# Izvršavanje funkcije
final_result = project_back_to_road(undist, warped, left_fit, right_fit, Minv)

# Prikaz i čuvanje
cv.imshow('Final Lane Detection', final_result)
cv.imwrite('result_files/final_lane_result.jpg', final_result)

print("Finalni rezultat je sačuvan u result_files/final_lane_result.jpg")
cv.waitKey(0)
cv.destroyAllWindows()