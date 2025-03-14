import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("WIN_20250207_17_44_19_Pro.mp4")

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and create grid points
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Define grid size
rows, cols = 12, 12  # Adjust for better spacing
h, w = old_gray.shape
step_x, step_y = w // cols, h // rows

# Generate fixed grid points
p0 = np.array([[j * step_x, i * step_y] for i in range(1, rows) for j in range(1, cols)], dtype=np.float32)
p0 = p0.reshape(-1, 1, 2)

# Store the original positions (fixed reference)
original_positions = p0.copy()

# Create a fading mask for drawing
mask = np.zeros_like(old_frame)

# **Dynamically adjust scale factor** (assuming a known FOV)
FOV_HORIZONTAL = 60  # Camera's horizontal field of view in degrees
FOV_VERTICAL = 40    # Vertical FOV (adjust as needed)
DEPTH = 1.0  # Assumed depth in meters

# Approximate conversion factor
PIXEL_TO_METER_X = (2 * DEPTH * np.tan(np.radians(FOV_HORIZONTAL / 2))) / w
PIXEL_TO_METER_Y = (2 * DEPTH * np.tan(np.radians(FOV_VERTICAL / 2))) / h

MIN_DRIFT_THRESHOLD = 0.002  # Ignore small movements under 2mm

def calculate_drift(good_new, good_orig):
    """
    Calculates the average drift in X and Y directions separately and converts to meters.
    Filters out outliers for more stable measurements.
    """
    good_new = good_new.reshape(-1, 2)
    good_orig = good_orig.reshape(-1, 2)

    # Calculate drift in pixels
    drift_x = (good_new[:, 0] - good_orig[:, 0]) * PIXEL_TO_METER_X
    drift_y = (good_new[:, 1] - good_orig[:, 1]) * PIXEL_TO_METER_Y

    # Remove outliers using median filtering
    median_x = np.median(drift_x)
    median_y = np.median(drift_y)

    # Consider only values within 1.5x median deviation
    valid_x = drift_x[np.abs(drift_x - median_x) < 1.5 * np.std(drift_x)]
    valid_y = drift_y[np.abs(drift_y - median_y) < 1.5 * np.std(drift_y)]

    avg_drift_x = np.mean(valid_x) if len(valid_x) > 0 else 0
    avg_drift_y = np.mean(valid_y) if len(valid_y) > 0 else 0

    return avg_drift_x, avg_drift_y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        st = st.flatten()
        valid_indices = np.where(st == 1)[0]  # Get indices of successfully tracked points
        good_new = p1[valid_indices].reshape(-1, 2)
        good_orig = original_positions[valid_indices].reshape(-1, 2)

        # Calculate drift in meters
        avg_drift_x, avg_drift_y = calculate_drift(good_new, good_orig)

        # **Reduce clutter by fading trails**
        mask = cv2.addWeighted(mask, 0.5, np.zeros_like(mask), 0.5, 0)

        # **Only draw significant movements**
        for new, orig in zip(good_new, good_orig):
            a, b = map(int, new.ravel())
            x0, y0 = map(int, orig.ravel())

            drift_magnitude = np.sqrt((a - x0) ** 2 + (b - y0) ** 2) * PIXEL_TO_METER_X
            if drift_magnitude > MIN_DRIFT_THRESHOLD:  # Only draw if drift is significant
                mask = cv2.line(mask, (x0, y0), (a, b), (0, 255, 0), 1)
                frame = cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)

        img = cv2.add(frame, mask)

        # Display the live drift values in meters
        drift_text = f"Drift X: {avg_drift_x:.4f} m, Drift Y: {avg_drift_y:.4f} m"
        cv2.putText(img, drift_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the result
        cv2.imshow('Live Drift Tracking (in meters)', img)

    # Update previous frame and points
    old_gray = frame_gray.copy()
    if len(valid_indices) > 0:
        p0 = p1[valid_indices].reshape(-1, 1, 2)
    else:
        p0 = original_positions

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
