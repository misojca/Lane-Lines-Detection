import cv2
import numpy as np

# ===============================
# Učitaj binarnu warped sliku
# ===============================
binary_warped = cv2.imread('results/warped_binary.jpg', cv2.IMREAD_GRAYSCALE)
binary_warped = binary_warped // 255

h, w = binary_warped.shape
ploty = np.linspace(0, h - 1, h)

# ===============================
# Polinomi (preuzeti iz z5.py)
# ===============================
left_fit = np.array([2.2e-4, -0.35, 300])
right_fit = np.array([2.1e-4, -0.30, 980])

left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# ===============================
# Konverzija piksel → metri
# ===============================
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700

left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

# ===============================
# Radius of curvature
# ===============================
y_eval = np.max(ploty) * ym_per_pix

left_curverad = (
    (1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5
) / np.abs(2*left_fit_cr[0])

right_curverad = (
    (1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5
) / np.abs(2*right_fit_cr[0])

# ===============================
# Offset vozila
# ===============================
lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
vehicle_center = w / 2
offset_meters = (vehicle_center - lane_center) * xm_per_pix

# ===============================
# Vizuelizacija
# ===============================
vis = np.dstack((binary_warped*255,)*3)

# Crtanje linija
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)

cv2.polylines(vis, pts_left, False, (255, 0, 0), 3)
cv2.polylines(vis, pts_right, False, (0, 0, 255), 3)

# ===============================
# Tekst za polinome
# ===============================
left_text = f"f_left(y) = {left_fit[0]:.2e}*y^2 + {left_fit[1]:.2f}*y + {left_fit[2]:.1f}"
right_text = f"f_right(y) = {right_fit[0]:.2e}*y^2 + {right_fit[1]:.2f}*y + {right_fit[2]:.1f}"

cv2.putText(vis, left_text, (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.putText(vis, right_text, (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Tekst za curvature i offset
cv2.putText(vis, f"Left curvature: {left_curverad:.1f} m", (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

cv2.putText(vis, f"Right curvature: {right_curverad:.1f} m", (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

cv2.putText(vis, f"Vehicle offset: {offset_meters:.2f} m", (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ===============================
# Prikaz i čuvanje
# ===============================
cv2.imshow('Lane Curvature & Position', vis)
cv2.imwrite('results/curvature_and_offset.jpg', vis)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ===============================
# Konzolni ispis
# ===============================
print(f"Left lane curvature: {left_curverad:.2f} m")
print(f"Right lane curvature: {right_curverad:.2f} m")
print(f"Vehicle offset from center: {offset_meters:.2f} m")
