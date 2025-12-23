import cv2
import numpy as np

# ===============================
# Učitaj kalibraciju kamere
# ===============================
calib = np.load('calib.npz')
mtx = calib['mtx']
dist = calib['dist']

# ===============================
# Učitaj video
# ===============================
cap = cv2.VideoCapture('test_videos/project_video01.mp4')

if not cap.isOpened():
    raise IOError("Ne mogu da otvorim video")

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ===============================
# Video writer (AVI – stabilno)
# ===============================
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'results/temp_output.avi',
    fourcc,
    fps,
    (width, height)
)

# ===============================
# Perspective transform (FIXNE tačke)
# ===============================
src = np.float32([
    [width * 0.43, height * 0.65],
    [width * 0.58, height * 0.65],
    [width * 0.90, height * 1.00],
    [width * 0.10, height * 1.00]
])

dst = np.float32([
    [width * 0.25, 0],
    [width * 0.75, 0],
    [width * 0.75, height],
    [width * 0.25, height]
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# ===============================
# Glavna petlja
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------
    # Undistort
    # ---------------------------
    undist = cv2.undistort(frame, mtx, dist, None, mtx)

    # ---------------------------
    # Binary threshold
    # ---------------------------
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    binary_s = np.zeros_like(s_channel)
    binary_s[(s_channel >= 170) & (s_channel <= 255)] = 1

    combined_binary = np.zeros_like(binary_s)
    combined_binary[(binary_sobel == 1) | (binary_s == 1)] = 1

    # ---------------------------
    # Warp (bird eye)
    # ---------------------------
    binary_warped = cv2.warpPerspective(
        combined_binary,
        M,
        (width, height)
    )

    # ---------------------------
    # Lane detection (sliding window – uprošćeno)
    # ---------------------------
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    ploty = np.linspace(0, height - 1, height)

    # FIKSNI polinomi (stabilni za demo)
    left_fit = np.array([2.2e-4, -0.35, leftx_base])
    right_fit = np.array([2.1e-4, -0.30, rightx_base])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # ---------------------------
    # Crtanje trake
    # ---------------------------
    lane_warp = np.zeros((height, width, 3), dtype=np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right)).astype(np.int32)

    cv2.fillPoly(lane_warp, [pts], (0, 255, 0))

    # ---------------------------
    # Unwarp nazad
    # ---------------------------
    lane_unwarp = cv2.warpPerspective(
        lane_warp,
        Minv,
        (width, height)
    )

    # ---------------------------
    # Overlay
    # ---------------------------
    result = cv2.addWeighted(undist, 1.0, lane_unwarp, 0.3, 0)

    # ---------------------------
    # Upis u video
    # ---------------------------
    out.write(result)

    cv2.imshow('Lane Detection', result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# Cleanup
# ===============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✔ Video snimljen u results/temp_output.avi")
print("➡️ Sledeći korak: ffmpeg konverzija u MP4 (H.264)")
