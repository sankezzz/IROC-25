import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks and calculate drift
        # Inside the loop where you draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Ensure coordinates are integers
            a, b = int(a), int(b)
            c, d = int(c), int(d)

            # Draw the line between the old and new position
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

            # Calculate drift values
            drift_x = a - c
            drift_y = b - d
            drift_text = f"Drift: ({drift_x:.2f}, {drift_y:.2f})"
            cv2.putText(frame, drift_text, (10, 30 + i * 20),
        cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)


        img = cv2.add(frame, mask)

        # Display the resulting frame
        cv2.imshow('Optical Flow Drift Tracking', img)

    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
