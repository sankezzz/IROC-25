import cv2
import numpy as np
import time
import threading
import keyboard 

class Drift():

    def __init__(self):    
        self.main_centroid = None  
        self.current_centroid = None 
        self.lock = threading.Lock()  
        self.lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.p0 = None
        self.newRefrence = False


    def capture_listener(self):

        while True:
            keyboard.wait('c')
            with self.lock:
                self.newRefrence = True
                
                print("Main frame captured with centroid:", self.main_centroid)
        
            time.sleep(0.2)

    def getFrames(self, cap):
        
        if not cap.isOpened():
            print("Failed to open video")
            exit()

        ret, frame = cap.read()

        if not ret:
            print("Failed to read video")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        return ret, frame
    
    def setRefrence(self, frame, mask=None):
        self.p0 = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        
    
    def showResult(self, frame, pt_list, FPS, centroid = None, drift = None):

        for new in pt_list:
            a, b = new.ravel()
            cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        cv2.circle(frame, centroid, 7, (0, 255, 255), -1)
        cv2.putText(frame, f"Frame Rate: {FPS} FPS", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if centroid:
            cv2.putText(frame, f"Centroid: {centroid}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if drift:
            cv2.putText(frame, drift, (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Centroid Tracking", frame)
        
    
    def compute(self):

        cap = cv2.VideoCapture(1)
        ret, old_frame = self.getFrames(cap)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        if p0 is not None:
            p0 = np.squeeze(p0)

        listener_thread = threading.Thread(target= self.capture_listener, daemon=True)
        listener_thread.start()

        prev_time = time.time()

        while cap.isOpened():
            drift = None
            centroid = None

            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # If the reference needs updating (i.e. immediately after pressing 'c')
            if self.newRefrence:
                if self. current_centroid is not None:
                    self.main_centroid = self.current_centroid
                # Update the reference frame exactly from the frame when 'c' was pressed.
                old_gray = frame_gray.copy()
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
                if p0 is not None:
                    p0 = np.squeeze(p0)
                # Reset the flag so we don't keep re-updating
                self.newRefrence = False
                continue

            # Compute optical flow using the previous gray frame and points.
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,
                                                    p0.reshape(-1, 1, 2),
                                                    None, **self.lk_params)

            if p1 is not None and st is not None:
                # Keep only valid points.
                good_new = p1[st.flatten() == 1]

                # If there's only one tracked point, ensure it is a 2D array with shape (1,2)
                if good_new.ndim == 1:
                    good_new = good_new.reshape(1, -1)
                elif good_new.ndim == 3 and good_new.shape[1] == 1:
                    good_new = good_new.reshape(-1, 2)

                # Calculate centroid of the tracked points.
                if len(good_new) > 0:
                    centroid_x = int(np.mean(good_new[:, 0]))
                    centroid_y = int(np.mean(good_new[:, 1]))
                    centroid = (centroid_x, centroid_y)

                    # Update the global current_centroid in a thread-safe manner.
                    with self.lock:
                        self.current_centroid = centroid

                    # Compute drift if a main centroid is available.
                    with self.lock:
                        if self.main_centroid is not None:
                            drift_x = centroid[0] - self.main_centroid[0]
                            drift_y = centroid[1] - self.main_centroid[1]
                            drift = f"Drift: ({drift_x}, {drift_y})"

                curr_time = time.time()
                frame_rate = round((1 / (curr_time - prev_time)) if curr_time - prev_time != 0 else 0, 2)
                prev_time = curr_time

                self.showResult(frame, good_new, frame_rate, centroid, drift)

                # Update p0 for the next iteration.
                p0 = good_new.copy() if good_new.size else p0

            old_gray = frame_gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    
    drift = Drift()
    drift.compute()

    cv2.destroyAllWindows()
