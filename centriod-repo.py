import cv2
import numpy as np
import time
import threading
import keyboard

class Drift():
    def __init__(self):
        self.fixed_centroid = None  # Centroid when 'c' is pressed
        self.current_centroid = None  # Live centroid
        self.lock = threading.Lock()  # Thread safety
        self.newReference = False  # Flag for updating fixed centroid

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.p0 = None

    def capture_listener(self):
        while True:
            keyboard.wait('c')
            with self.lock:
                self.newReference = True  # Trigger reference update
            time.sleep(0.2)  # Prevent multiple triggers

    def getFrames(self, cap):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        return ret, frame

    def showResult(self, frame, pt_list, FPS, current_centroid=None, fixed_centroid=None, drift=None):
        for new in pt_list:
            a, b = new.ravel()
            cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)  # Red dots for tracked points

        if current_centroid:
            cv2.circle(frame, current_centroid, 7, (0, 255, 255), -1)  # Yellow for current centroid

        if fixed_centroid:
            cv2.circle(frame, fixed_centroid, 7, (255, 0, 0), -1)  # Blue for fixed centroid

        cv2.putText(frame, f"Frame Rate: {FPS} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_centroid:
            cv2.putText(frame, f"Current Centroid: {current_centroid}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if fixed_centroid:
            cv2.putText(frame, f"Fixed Centroid: {fixed_centroid}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if drift:
            cv2.putText(frame, drift, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Centroid Tracking", frame)

    def compute(self):
        cap = cv2.VideoCapture("drone pov.mp4")
        ret, old_frame = self.getFrames(cap)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)

        if p0 is not None:
            p0 = np.squeeze(p0)

        listener_thread = threading.Thread(target=self.capture_listener, daemon=True)
        listener_thread.start()

        prev_time = time.time()

        while cap.isOpened():
            drift_text = None
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Update fixed centroid exactly when 'c' is pressed
            if self.newReference:
                with self.lock:
                    if self.current_centroid:
                        self.fixed_centroid = self.current_centroid  # Capture current centroid as fixed
                    old_gray = frame_gray.copy()
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
                    if p0 is not None:
                        p0 = np.squeeze(p0)
                    self.newReference = False  # Reset flag
                continue

            # Optical flow tracking
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                                    p0.reshape(-1, 1, 2),
                                                    None, **self.lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st.flatten() == 1]

                if good_new.ndim == 1:
                    good_new = good_new.reshape(1, -1)
                elif good_new.ndim == 3 and good_new.shape[1] == 1:
                    good_new = good_new.reshape(-1, 2)

                # Compute current centroid
                if len(good_new) > 0:
                    centroid_x = int(np.mean(good_new[:, 0]))
                    centroid_y = int(np.mean(good_new[:, 1]))
                    self.current_centroid = (centroid_x, centroid_y)

                    # Compute drift if fixed centroid exists
                    with self.lock:
                        if self.fixed_centroid:
                            drift_x = self.current_centroid[0] - self.fixed_centroid[0]
                            drift_y = self.current_centroid[1] - self.fixed_centroid[1]
                            drift_text = f"Drift: ({drift_x}, {drift_y})"

                curr_time = time.time()
                frame_rate = round((1 / (curr_time - prev_time)) if curr_time - prev_time != 0 else 0, 2)
                prev_time = curr_time

                self.showResult(frame, good_new, frame_rate, self.current_centroid, self.fixed_centroid, drift_text)

                # Update p0 for next iteration
                p0 = good_new.copy() if good_new.size else p0

            old_gray = frame_gray.copy()

            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    drift = Drift()
    drift.compute()
    cv2.destroyAllWindows()
