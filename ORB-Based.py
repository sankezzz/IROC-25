import cv2
import numpy as np

# Initialize ORB detector
orb = cv2.ORB_create()

# Open webcam
cap = cv2.VideoCapture("drone pov.mp4")

# Set a fixed display size (adjustable)
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 480

initial_frame = None
initial_gray = None
initial_keypoints = None
initial_descriptors = None
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("Press SPACE to capture the initial frame...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Show live feed before capturing initial frame
    if initial_frame is None:
        cv2.putText(frame, "Press SPACE to set the reference frame", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ORB Camera Displacement", frame)

        # Capture initial frame when SPACE is pressed
        if cv2.waitKey(10) & 0xFF == ord(' '):
            initial_frame = frame.copy()
            initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
            initial_keypoints, initial_descriptors = orb.detectAndCompute(initial_gray, None)
            print("Initial frame captured! Now tracking displacement...")
        continue  # Wait for initial frame to be set

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the current frame
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is not None and initial_descriptors is not None:
        # Match descriptors using Brute Force Matcher
        matches = bf.match(initial_descriptors, descriptors)

        # Sort matches based on distance (lower = better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract keypoints coordinates
        src_pts = np.float32([initial_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate transformation matrix (Homography)
        if len(matches) > 10:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Extract translation (displacement)
                dx = H[0, 2]
                dy = H[1, 2]

                # Set depth (dz) as a constant
                dz = 100  # Fixed depth

                # Display displacement
                text = f"Displacement: dx={dx:.2f}, dy={dy:.2f}, dz={dz}"
                cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw matches for visualization
        frame_with_keypoints = cv2.drawMatches(initial_frame, initial_keypoints, frame, keypoints, matches[:20], None, flags=2)

        # Resize final output to fit screen
        frame_with_keypoints = cv2.resize(frame_with_keypoints, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Show the frame
        cv2.imshow("ORB Camera Displacement", frame_with_keypoints)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
