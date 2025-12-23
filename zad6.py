import numpy as np
import cv2 as cv

# 1. Učitavanje podataka
poly_data = np.load('result_files/poly_fit.npz')
left_fit = poly_data['left_fit']
right_fit = poly_data['right_fit']

binary_warped = cv.imread('result_files/warped_binary_image.jpg', 0)

def calculate_metrics(img_shape, left_fit, right_fit):
    # Konverzija piksel -> metar
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    y_eval = img_shape[0] - 1
    
    # Generisanje y vrednosti i x koordinata polinoma
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Radijus krivine u realnom prostoru (metri)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    l_rad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    r_rad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Pozicija vozila u odnosu na centar
    car_pos = img_shape[1] / 2
    lane_center = (leftx[-1] + rightx[-1]) / 2
    offset = (lane_center - car_pos) * xm_per_pix
    
    return l_rad, r_rad, offset, ploty, leftx, rightx

# Izračunavanje metrika i dobijanje tačaka za crtanje
l_rad, r_rad, offset, ploty, leftx, rightx = calculate_metrics(binary_warped.shape, left_fit, right_fit)

# 2. VIZUELIZACIJA
# Kreiramo sliku u boji za iscrtavanje
out_img = np.dstack((binary_warped, binary_warped, binary_warped))

# Iscrtavanje leve i desne linije trake (žuta boja)
# Pripremamo tačke za cv.polylines
left_pts = np.array([np.transpose(np.vstack([leftx, ploty]))], np.int32)
right_pts = np.array([np.transpose(np.vstack([rightx, ploty]))], np.int32)

cv.polylines(out_img, [left_pts], False, (0, 255, 255), 5)  # Žuta leva
cv.polylines(out_img, [right_pts], False, (0, 255, 255), 5) # Žuta desna

# Priprema teksta jednačina
text_left = f"f_left(y) = {left_fit[0]:.2e}*y^2 + {left_fit[1]:.2f}*y + {left_fit[2]:.2f}"
text_right = f"f_right(y) = {right_fit[0]:.2e}*y^2 + {right_fit[1]:.2f}*y + {right_fit[2]:.2f}"
text_curv = f"Radius: {(l_rad+r_rad)/2:.2f}m | Offset: {offset:.2f}m"

# Ispisivanje teksta na sliku
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(out_img, text_left, (50, 50), font, 0.8, (0, 255, 0), 2)
cv.putText(out_img, text_right, (50, 100), font, 0.8, (0, 255, 0), 2)
cv.putText(out_img, text_curv, (50, 150), font, 0.8, (255, 255, 255), 2)

# Prikazivanje i čuvanje rezultata
cv.imshow('Polynomial Fit and Metrics', out_img)
cv.imwrite('result_files/polynomial_and_metrics.jpg', out_img)

print("-" * 50)
print(text_left)
print(text_right)
print(f"Radijus zakrivljenosti: {(l_rad+r_rad)/2:.2f}m")
print(f"Udaljenost od centra: {offset:.2f}m")
print("-" * 50)

cv.waitKey(0)
cv.destroyAllWindows()