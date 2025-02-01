import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("drone pov.mp4")

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Feature detection parameters
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Read the first frame and detect features
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Real-world scale and frame rate (adjust based on calibration)
scale_factor = 2  # meters per pixel
frame_rate = 30  # FPS

# Variables to store captured frame information
capture_position = None
captured_coordinates = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        # Keep only valid points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate motion vectors
        motion_vectors = good_new - good_old
        distances = np.linalg.norm(motion_vectors, axis=1)
        avg_distance = np.mean(distances) if len(distances) > 0 else 0

        # Convert to real-world units
        real_distance = avg_distance * scale_factor
        speed = real_distance * frame_rate  # meters per second

        # Drift Calculation: Difference in motion over time
        drift = np.std(distances) if len(distances) > 0 else 0  # Standard deviation of motion

        # Draw tracking lines
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Capture the reference frame's position when spacebar is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Press SPACE to capture coordinates
            capture_position = good_new.copy()  # Save current tracking points
            captured_coordinates = [(int(x), int(y)) for x, y in capture_position]  # Store as integer values

        # Display speed, drift, and captured frame coordinates
        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Drift: {drift:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if captured_coordinates:
            cv2.putText(frame, f"Captured Coords: {captured_coordinates[0]}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Update frame data for next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Display the live feed
    cv2.imshow("Live Speed & Drift Feed", frame)

    # Exit loop on pressing 'q'
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
