import cv2
import numpy as np
import matplotlib.pyplot as plt

# Učitaj warped binarnu sliku (iz prethodnog koraka)
binary_warped = cv2.imread('results/warped_binary.jpg', cv2.IMREAD_GRAYSCALE)

# Pretvori u binarnu (0 ili 1)
binary_warped = binary_warped // 255

# Visina i širina slike
h, w = binary_warped.shape

# ===============================
# 1. Histogram donje polovine slike
# ===============================
histogram = np.sum(binary_warped[h//2:, :], axis=0)

# Pronađi početne tačke leve i desne linije
midpoint = w // 2
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# ===============================
# 2. Sliding windows parametri
# ===============================
nwindows = 9
window_height = h // nwindows
margin = 100
minpix = 50

# Pozicije svih belih piksela
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base

left_lane_inds = []
right_lane_inds = []

# ===============================
# 3. Sliding windows
# ===============================
for window in range(nwindows):
    win_y_low = h - (window + 1) * window_height
    win_y_high = h - window * window_height

    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    good_left_inds = (
        (nonzeroy >= win_y_low) &
        (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &
        (nonzerox < win_xleft_high)
    ).nonzero()[0]

    good_right_inds = (
        (nonzeroy >= win_y_low) &
        (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &
        (nonzerox < win_xright_high)
    ).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))

    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))

# Spoji indekse
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# ===============================
# 4. Izvuci piksele leve i desne linije
# ===============================
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# ===============================
# 5. Fit polinoma 2. stepena
# ===============================
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

print("Left lane polynomial:", left_fit)
print("Right lane polynomial:", right_fit)

# ===============================
# 6. Vizualizacija
# ===============================
ploty = np.linspace(0, h - 1, h)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

plt.figure(figsize=(10, 6))
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow', linewidth=2)
plt.plot(right_fitx, ploty, color='yellow', linewidth=2)
plt.title("Lane Line Pixel Detection and Polynomial Fit")
plt.gca().invert_yaxis()
plt.show()
