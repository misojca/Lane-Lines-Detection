import cv2
import numpy as np

# ===============================
# Učitaj originalnu sliku puta
# ===============================
original = cv2.imread('test_images/test1.jpg')

# ===============================
# Učitaj warped binarnu sliku
# ===============================
binary_warped = cv2.imread('results/warped_binary.jpg', cv2.IMREAD_GRAYSCALE)
binary_warped = binary_warped // 255

h, w = binary_warped.shape
ploty = np.linspace(0, h - 1, h)

# ===============================
# Polinomi (iz z5 / z6)
# ===============================
left_fit = np.array([2.2e-4, -0.35, 300])
right_fit = np.array([2.1e-4, -0.30, 980])

left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# ===============================
# Napravi masku za traku
# ===============================
lane_warp = np.zeros((h, w, 3), dtype=np.uint8)

pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

pts = np.hstack((pts_left, pts_right)).astype(np.int32)

cv2.fillPoly(lane_warp, [pts], (0, 255, 0))

# ===============================
# Perspective transform (INVERSE)
# ===============================
src = np.float32([
    [w * 0.43, h * 0.65],
    [w * 0.58, h * 0.65],
    [w * 0.90, h * 1.00],
    [w * 0.10, h * 1.00]
])

dst = np.float32([
    [w * 0.25, 0],
    [w * 0.75, 0],
    [w * 0.75, h],
    [w * 0.25, h]
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Warp nazad na original
lane_unwarp = cv2.warpPerspective(
    lane_warp, Minv, (original.shape[1], original.shape[0])
)

# ===============================
# Overlay na originalnu sliku
# ===============================
result = cv2.addWeighted(original, 1, lane_unwarp, 0.3, 0)

# ===============================
# Prikaz i snimanje
# ===============================
cv2.imshow('Final Lane Detection', result)
cv2.imwrite('results/final_lane_overlay.jpg', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
