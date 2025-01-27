import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Parameters for optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Get the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial features to track
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Assume scale factor for real-world conversion (meters per pixel)
scale_factor = 0.02  # Example: 0.02 meters per pixel (calibrate for your setup)
frame_rate = 30  # Frames per second of the camera (adjust for your camera)

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Compute motion vectors and distances
        motion_vectors = good_new - good_old
        distances = np.linalg.norm(motion_vectors, axis=1)  # Pixel distances
        avg_distance = np.mean(distances) if len(distances) > 0 else 0

        # Convert pixel movement to real-world distance
        real_distance = avg_distance * scale_factor  # Distance in meters

        # Calculate speed (meters per second)
        speed = real_distance * frame_rate  # Real distance per second

        # Display the speed on the frame
        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Update points for the next frame
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)


        # Update the old points and frame
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        # Re-detect features if lost
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()

    # Show the live feed with speed overlay
    cv2.imshow("Live Speed Feed", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
