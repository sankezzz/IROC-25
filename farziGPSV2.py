import cv2
import numpy as np
import time
import threading
import keyboard 

class Drift():

    def __init__(self):    

        self.lock = threading.Lock()  

        self.min_feature          = 10
        self.alpha                = 0.2
        self.lk_params            = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params       = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.camera_mat           = np.array([[1.42031767e+03, 0.00000000e+00, 6.26243614e+02],# Logitech C270 camera
                                              [0.00000000e+00, 1.41976083e+03, 3.86318058e+02],
                                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.dist_coeffs          = np.array([[ 0.05568459, 0.56235417, -0.00208022, -0.00117773, -0.78622635]])
        self.filtered_centroid    = None
        self.main_centroid        = None  
        self.current_centroid     = None 
        self.current_points       = None
        self.p0                   = None
        self.newRefrence          = False
        self.total_drift          = [0, 0]
        self.drift_before_reset   = 30
        self.frame_rate           = 20

        # Initialize Kalman Filter:
        # State: [drift_x, drift_y, drift_dx, drift_dy, acc_x, acc_y]
        # Measurement: [drift_x, drift_y]
        # assumption const accln
        self.kf = cv2.KalmanFilter(6, 2)
        dt = 1/ self.frame_rate  # time step (assumed 1 frame; adjust if needed)
        self.kf.transitionMatrix = np.array([[1, 0, dt,  0, 0.5*dt**2,         0],
                                             [0, 1,  0, dt,         0, 0.5*dt**2],
                                             [0, 0,  1,  0,        dt,         0],
                                             [0, 0,  0,  1,         0,        dt],
                                             [0, 0,  0,  0,         1,         0],
                                             [0, 0,  0,  0,         0,         1]], dtype=np.float32)

        self.kf.measurementMatrix   = np.array([[1, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 20
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 30e-1
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self.kf.statePost           = np.zeros((6, 1), dtype=np.float32)


    def capture_listener(self):

        while True:
            keyboard.wait('c')
            with self.lock:
                self.newRefrence = True
                self.total_drift = [0, 0]
                print("Main frame captured with centroid:", self.main_centroid)
        
            #time.sleep(0.2)

    def getFrames(self, cap):
        
        if not cap.isOpened():
            print("Failed to open video")
            exit()

        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        #frame = cv2.undistort(frame, self.camera_mat, self.dist_coeffs, None, self.camera_mat)
        if not ret:
            print("Failed to read video")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        return ret, frame
    
    def setRefrence(self, frame, mask=None):
        self.p0 = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

    def showResult(self, frame, pt_list, FPS, drift = None, geo_coord = None):
        # Display the drift (relative to reference) on the frame.
        if pt_list is not None:
            for new in pt_list:
                a, b = new.ravel()
                cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"FPS: {FPS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if drift:
            cv2.putText(frame, f"Drift(m): {float(drift[0]), float(drift[1])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if geo_coord:
            cv2.putText(frame, f"Lat/Long: {float(geo_coord[0]), float(geo_coord[0])}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Drift Tracking", frame)

    def cumulate_drift(self, drift):

        drift_magnitude = np.linalg.norm(drift) 
        threshold = self.drift_before_reset  

        if drift_magnitude > threshold:
            self.total_drift[0] += drift[0]
            self.total_drift[1] += drift[1]

            self.kf.errorCovPost *= 10  
            
            self.newRefrence = True
            return True
        else:
            self.kf.errorCovPost *= 0.9 
            self.kf.errorCovPost = np.clip(self.kf.errorCovPost, 1e-2, 1e3)  

        return False
    
    def drift_to_geo_coord(self, drift, altitude):
        
        scale_factor = altitude / self.camera_mat[0][0]
        drift_x_m = scale_factor * drift[0]
        drift_y_m = scale_factor * drift[1]

        lon = drift_y_m / 111320.0
        lat = drift_x_m / 111320.0

        return (lat, lon), (drift_x_m, drift_y_m)
   
    def compute(self):

        cap = cv2.VideoCapture("C:/Users/SHLOAK/Pictures/Camera Roll/WIN_20250207_17_47_48_Pro.mp4")
        ret, init_frame = self.getFrames(cap)
        good_new = None

        if not ret:
            print("Failed to open video")
            return

        init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        # Initially, use the first frame as reference.
        self.ref_gray = init_gray.copy()
        p0 = cv2.goodFeaturesToTrack(self.ref_gray, mask=None, **self.feature_params)
        if p0 is not None:
            self.p_ref = np.squeeze(p0)
        else:
            print("No features found in the reference frame.")
            return

        # Start listener thread to update reference on demand.
        listener_thread = threading.Thread(target=self.capture_listener, daemon=True)
        listener_thread.start()
        prev_time = time.time()

        self.kf.statePost = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float32)

        while cap.isOpened():

            measured_drift = None  # Raw measurement from optical flow
            ret, frame = self.getFrames(cap)
            if not ret:
                break

            dt = 1/self.frame_rate
            dt = max(0.001, min(dt, 1.0))  

            self.kf.transitionMatrix = np.array([[1, 0, dt,  0, 0.5*dt**2,         0],
                                             [0, 1,  0, dt,         0, 0.5*dt**2],
                                             [0, 0,  1,  0,        dt,         0],
                                             [0, 0,  0,  1,         0,        dt],
                                             [0, 0,  0,  0,         1,         0],
                                             [0, 0,  0,  0,         0,         1]], dtype=np.float32)


            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.newRefrence:
                self.ref_gray = frame_gray.copy()
                p_new = cv2.goodFeaturesToTrack(self.ref_gray, mask=None, **self.feature_params)
                if p_new is not None and len(p_new) >= self.min_feature:
                    self.p_ref = np.squeeze(p_new)
                    print("Reference frame updated.")
                else:
                    print("Not enough features for reference frame. Keeping the old reference.")
                self.newRefrence = False

            # Compute optical flow from the fixed reference frame to the current frame.
            if self.p_ref is not None and len(self.p_ref) >= self.min_feature:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.ref_gray, frame_gray,
                                                       self.p_ref.reshape(-1, 1, 2),
                                                       None, **self.lk_params)

                if p1 is not None and st is not None:
                    good_new = p1[st.flatten() == 1]
                    good_ref = self.p_ref[st.flatten() == 1]

                    # Ensure we have enough points for RANSAC (typically at least 4)
                    if len(good_new) >= 4:
                        M, inliers = cv2.estimateAffinePartial2D(good_ref, good_new, method=cv2.RANSAC)
                        if M is not None:
                            # M has form: [ [a, b, tx],
                            #               [c, d, ty] ]
                            measured_drift = (M[0, 2], M[1, 2])

            # Kalman Filter update: if we have a new measurement, update the filter.
            if measured_drift is not None:
                measurement = np.array([[np.float32(measured_drift[0])],
                                        [np.float32(measured_drift[1])]])
                self.kf.correct(measurement)
            # Predict the next state
            prediction = self.kf.predict()
            filtered_drift = [int(prediction[0]), int(prediction[1])]

            self.cumulate_drift(filtered_drift)

            filtered_drift[0] += self.total_drift[0]
            filtered_drift[1] += self.total_drift[1]
            geo_coord, drift_meter = self.drift_to_geo_coord(filtered_drift, 5)

            curr_time = time.time()
            self.frame_rate = round(1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0, 2)
            prev_time = curr_time

            self.showResult(frame, good_new, self.frame_rate, drift_meter, geo_coord)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        



if __name__ == "__main__":
    
    drift = Drift()
    drift.compute()

    cv2.destroyAllWindows()