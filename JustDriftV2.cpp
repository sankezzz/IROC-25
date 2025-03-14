#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <mutex>
#include <cmath>
#include <chrono>
#include <algorithm>

using namespace cv;
using namespace std;


//-------------------------------------------
// Drift class: Handles video capture, optical flow, drift computation, and Kalman filtering
//-------------------------------------------
class Drift {
public:
    // Thread synchronization
    mutex mtx;
    mutex GPSmtx;
    
    // Frame dimensions
    int frame_width = 1280;
    int frame_height = 720;
    
    // Grid parameters
    int total_grid_points = 49;
    vector<Point2f> grid_points;
    
    // Optical flow parameters
    int min_feature = 10;
    Size winSize = Size(15, 15);
    int maxLevel = 2;
    TermCriteria termcrit = TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 0.03);
    int maxCorners = 50;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;
    
    // Camera calibration (Logitech C270)
    Mat camera_mat;
    Mat dist_coeffs;
    Mat prevGray;
    
    // Reference frame and feature points for optical flow
    Mat ref_gray;
    vector<Point2f> p_ref;
    
    // Flags and state
    bool newReference = false;
    Mat latest_frame;
    bool isLatest = false;
    
    // Drone and sensor state
    float altitude = 0.0f; // in meters
    float roll = 0.0f;     // in radians
    float pitch = 0.0f;
    float yaw = 0.0f;
    float init_yaw = 0.0f;
    
    // Drift accumulation
    float total_drift[2] = {0.0f, 0.0f};
    float drift_before_reset = 0.02f; // meters
    float alt_before_reset = 0.5f;      // meters
    float prev_alt = 0.0f;
    float frame_rate = 20.0f;
    float x = 0.0;
    float y = 0.0;
    float prevDriftx = 0.0;
    float prevDrifty = 0.0;   
    
    // Kalman Filter: state dimension 6, measurement dimension 2.
    KalmanFilter kf;
    
    // Constructor: initialize matrices, grid points, and Kalman filter.
    Drift() : kf(6, 2, 0) {
        // Set up camera matrix and distortion coefficients
        camera_mat = (Mat_<double>(3,3) << 1.42031767e+03, 0, 6.26243614e+02,
                                           0, 1.41976083e+03, 3.86318058e+02,
                                           0, 0, 1);
        dist_coeffs = (Mat_<double>(1,5) << 0.05568459, 0.56235417, -0.00208022, -0.00117773, -0.78622635);
                                           
        // Compute grid points
        grid_points = getGridPoints();
        p_ref = grid_points;  // initial reference features
        
        // Initialize Kalman filter.
        float dt = 1.0f / frame_rate;
        kf.transitionMatrix = (Mat_<float>(6, 6) <<
            1, 0, dt, 0, 0.5f * dt * dt, 0,
            0, 1, 0, dt, 0, 0.5f * dt * dt,
            0, 0, 1, 0, dt, 0,
            0, 0, 0, 1, 0, dt,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);
            
        kf.measurementMatrix = (Mat_<float>(2, 6) <<
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0);
            
        kf.processNoiseCov = Mat::eye(6, 6, CV_32F) * 0.2;//0.75
        kf.measurementNoiseCov = Mat::eye(2, 2, CV_32F) * 0.0006f;//0.0004
        kf.errorCovPost = Mat::eye(6, 6, CV_32F);
        kf.statePost = Mat::zeros(6, 1, CV_32F);
    }
    
    // Generate grid points
    vector<Point2f> getGridPoints() {
        vector<Point2f> pts;
        int grid_size = (int)ceil(sqrt((double)total_grid_points));
        vector<float> x_pts, y_pts;
        float x_start = 30.0f, x_end = frame_width - 30.0f;
        float y_start = 30.0f, y_end = frame_height - 30.0f;
        if (grid_size > 1) {
            float x_step = (x_end - x_start) / (grid_size - 1);
            float y_step = (y_end - y_start) / (grid_size - 1);
            for (int i = 0; i < grid_size; i++) {
                x_pts.push_back(x_start + i * x_step);
                y_pts.push_back(y_start + i * y_step);
            }
        } else {
            x_pts.push_back((x_start + x_end) / 2.0f);
            y_pts.push_back((y_start + y_end) / 2.0f);
        }
        // Create meshgrid and take the first total_grid_points points.
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                pts.push_back(Point2f(x_pts[j], y_pts[i]));
                if (pts.size() >= (size_t)total_grid_points)
                    return pts;
            }
        }
        return pts;
    }
    
    // Display results on the frame (drawing features, overlaying text, etc.)
    void showResult(Mat &frame, const vector<Point2f>& pt_list, double FPS, const Point2f* drift) {
        for (auto pt : pt_list) {
            circle(frame, pt, 3, Scalar(255, 0, 0), -1);
        }
        // Draw grid points in red.
        for (auto pt : grid_points) {
            circle(frame, pt, 3, Scalar(0, 0, 255), -1);
        }
        // Overlay text.
        string fpsText = "FPS: " + to_string(FPS);
        putText(frame, fpsText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        string attText = "Alt (m): " + to_string(altitude) +
                         ", Roll: " + to_string(roll) +
                         ", Pit: " + to_string(pitch) +
                         ", Yaw: " + to_string(yaw);
        putText(frame, attText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        if (drift) {
            string driftText = "Drift (m): (" +
                to_string(round(drift->x * 1000) / 1000.0) + ", " +
                to_string(round(drift->y * 1000) / 1000.0) + ")";
            putText(frame, driftText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        imshow("Drift Tracking", frame);

    }
    
    // Compute a rotation (homography) matrix based on the current yaw, pitch, and roll.
    Mat getHomographyMat() {
        // Rotation about Z (yaw)
        Mat Rz = (Mat_<double>(3, 3) <<
            cos(yaw), -sin(yaw), 0,
            sin(yaw),  cos(yaw), 0,
            0,         0,        1);
        // Rotation about X (roll)
        Mat Rx = (Mat_<double>(3, 3) <<
            1, 0,         0,
            0, cos(roll), -sin(roll),
            0, sin(roll),  cos(roll));
        // Rotation about Y (pitch)
        Mat Ry = (Mat_<double>(3, 3) <<
            cos(pitch), 0, sin(pitch),
            0,          1, 0,
            -sin(pitch),0, cos(pitch));
        // Combined rotation: Rz * Ry * Rx
        Mat H = Rz * Ry * Rx;
        return H;
    }
    
    // Remove rotation from a measured drift vector.
    Point3f removeRotation(const Point2f &drift) {
        // Create a 3x1 vector from drift (with a homogeneous coordinate 1)
        Mat measured = (Mat_<double>(3, 1) << drift.x, drift.y, 1);
        Mat H = getHomographyMat();
        Mat H_inv = H.inv();
        Mat result = H_inv * measured;
        return Point3f((float)result.at<double>(0, 0),
                       (float)result.at<double>(1, 0),
                       (float)result.at<double>(2, 0));
    }
    
    // Accumulate drift if the magnitude exceeds thresholds.
    bool cumulateDrift(const Point2f &drift) {
        float drift_magnitude = sqrt(drift.x * drift.x + drift.y * drift.y);
        if (drift_magnitude > drift_before_reset || fabs(altitude - prev_alt) > alt_before_reset) {
            total_drift[0] += drift.x;
            total_drift[1] += drift.y;
            prev_alt = altitude;
            newReference = true;
            return true;
        }
        return false;
    }


    void frameReader(VideoCapture &cap) {
		const char* shm_name = "/shm_Attitude";
		const char* sem_name = "/sem_Attitude";

		// Open the named semaphore.
		sem_t* semaphore = sem_open(sem_name, 0);
		if (semaphore == SEM_FAILED) {
			std::cerr << "Error opening Drone Attitude semaphore!\n";
			return;
		}
		// Open shared memory.
		int fd = shm_open(shm_name, O_RDWR, 0666);
		if (fd == -1) {
			std::cerr << "Error opening Attitude shared memory\n";
			sem_close(semaphore);
			return;
		}
		char* data = (char*)mmap(nullptr, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		//close(fd);

		// Continuously capture frames.
		while (cap.isOpened()) {
			Mat frame;
			cap >> frame;
			if (frame.empty()) {
				cout << "Error: Frame not read!" << endl;
				break;
			}

			{
				lock_guard<mutex> guard(mtx);
				resize(frame, latest_frame, Size(frame_width, frame_height));
				isLatest = true;
			}

			if (sem_trywait(semaphore) == 0) {
                //cout << "Semaphore acquired. Raw data: " << data << endl;
                char* token = strtok(data, ",");
                float attitude_vals[4] = {0};
                int i = 0;
                while (token != nullptr && i < 4) {
                    attitude_vals[i++] = std::stof(token);
                    token = strtok(nullptr, ",");
                }
                {
                    lock_guard<mutex> guard(mtx);
                    altitude = attitude_vals[0];
                    roll     = attitude_vals[1];
                    pitch    = attitude_vals[2];
                    yaw      = attitude_vals[3] - init_yaw;
                }
            } else {
                cout << "Semaphore not available." << endl;
            }
            
			this_thread::sleep_for(chrono::milliseconds(1));
		}

		sem_close(semaphore);
		munmap(data, 1024);
    }

    void DriftWriter(double drift_x, double drift_y) {
        const char* shm_name = "/shm_Drift";
        const char* sem_name = "/sem_Drift";

        static sem_t* Driftsemaphore = sem_open(sem_name, 0);
        if (Driftsemaphore == SEM_FAILED) {
            cerr << "Error opening Drone Drift semaphore!\n";
            return;
        }
 
        static int fd = shm_open(shm_name, O_RDWR, 0666);
        if (fd == -1) {
            cerr << "Error opening Drift shared memory\n";
            sem_close(Driftsemaphore);
            return;
        }
        static char*
         Driftdata = (char*)mmap(nullptr, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        //close(fd);
    
        if (Driftdata == MAP_FAILED) {
            cerr << "Error mapping Drift shared memory\n";
            sem_close(Driftsemaphore);
            return;
        }

        {
            lock_guard<mutex> guard(Driftmtx);
            memset(Driftdata, 0, 1024);
            stringstream ss;
            x = drift_x;
            y = drift_y;
            ss << x << "," << y;
            string token = ss.str();
            snprintf(Driftdata, 1024, "%s", token.c_str());
        }

        sem_post(Driftsemaphore);
        
    }

    
    // Main computation loop.
    void compute() {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cout << "Error: Unable to open camera!" << endl;
            return;
        }
        
        // Start a separate thread to continuously read frames.
        thread frameThread(&Drift::frameReader, this, ref(cap));
        frameThread.detach();
        
        // Wait until the first frame is available.
        cout << "Waiting for first frame..." << endl;
        while (true) {
            lock_guard<mutex> guard(mtx);
            if (!latest_frame.empty()) break;
            this_thread::sleep_for(chrono::milliseconds(1));
        }
        {
            lock_guard<mutex> guard(mtx);
            cvtColor(latest_frame, ref_gray, COLOR_BGR2GRAY);
        }
        
        // Initialize Kalman filter state.
        kf.statePost = Mat::zeros(6, 1, CV_32F);
        double prev_time = (double)getTickCount();
        vector<Point2f> good_new;
        init_yaw = yaw;
        
        // Assume prevGray is declared as a Mat at class scope (or before the loop)
// For example: Mat prevGray;

		Mat prevGray, prevSmallGray; // Declare outside the loop
		float scaleFactor = 0.5;       // Scale factor for downsampling

		while (cap.isOpened()) {
            Mat frame, frame_gray;
            {
                lock_guard<mutex> guard(mtx);
                if (latest_frame.empty() || !isLatest) continue;
                frame = latest_frame.clone();
                isLatest = false;
            }
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);
           
            
            // Update Kalman filter transition matrix based on elapsed time.
            double curr_time = (double)getTickCount();
            double elapsed = (curr_time - prev_time) / getTickFrequency();
            prev_time = curr_time;
            float dt = (float)max(0.001, min(elapsed, 1.0));
            kf.transitionMatrix = (Mat_<float>(6, 6) <<
                1, 0, dt, 0, 0.5f * dt * dt, 0,
                0, 1, 0, dt, 0, 0.5f * dt * dt,
                0, 0, 1, 0, dt, 0,
                0, 0, 0, 1, 0, dt,
                0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1);
            
            // If a new reference frame was requested, update the reference image and features.
            if (newReference) {
                ref_gray = frame_gray.clone();
                p_ref = grid_points;
                newReference = false;
            }
            
            
            // Compute optical flow if there are enough features.
            Point2f measured_drift(0, 0);
            vector<Point2f> filtered_good_new;
            vector<Point2f> filtered_good_ref;
            bool validFlow = false;
            if (!p_ref.empty() && p_ref.size() >= (size_t)min_feature) {
                vector<uchar> status;
                vector<float> err;
                vector<Point2f> p1;
                calcOpticalFlowPyrLK(ref_gray, frame_gray, p_ref, p1, status, err, winSize, maxLevel, termcrit);
                vector<Point2f> good_ref;
                good_new.clear();
                for (size_t i = 0; i < p1.size(); i++) {
                    if (status[i]) {
                        good_new.push_back(p1[i]);
                        good_ref.push_back(p_ref[i]);
                    }
                }

                vector<Point2f> driftVec;
                for (size_t i = 0; i < good_new.size(); i++) {
                    driftVec.push_back(good_new[i] - good_ref[i]);
                }
                //compute median.
                auto median = [](vector<float>& v) -> float {
                    sort(v.begin(), v.end());
                    size_t n = v.size();
                    if (n % 2 == 0)
                        return (v[n/2 - 1] + v[n/2]) / 2.0f;
                    else
                        return v[n/2];
                };
                vector<float> dx, dy;
                for (const auto &d : driftVec) {
                    dx.push_back(d.x);
                    dy.push_back(d.y);
                }

                // Calculate median for x and y.
                float med_dx = median(dx);
                float med_dy = median(dy);

                // Compute median absolute deviation (MAD) for x and y.
                vector<float> abs_dx, abs_dy;
                for (size_t i = 0; i < driftVec.size(); i++) {
                    abs_dx.push_back(fabs(driftVec[i].x - med_dx));
                    abs_dy.push_back(fabs(driftVec[i].y - med_dy));
                }
                float mad_dx = median(abs_dx);
                float mad_dy = median(abs_dy);

                // Set thresholds (adjust multiplier as needed)
                float thresh_multiplier = 1.5f;
                float thresh_x = thresh_multiplier * mad_dx;
                float thresh_y = thresh_multiplier * mad_dy;

                // Filter out outliers based on the threshold.
                
                for (size_t i = 0; i < driftVec.size(); i++) {
                    if (fabs(driftVec[i].x - med_dx) <= thresh_x &&
                        fabs(driftVec[i].y - med_dy) <= thresh_y) {
                        filtered_good_new.push_back(good_new[i]);
                        filtered_good_ref.push_back(good_ref[i]);
                    }
                }

                if (filtered_good_new.size() < 3) {
                    std::cerr << "Not enough points to estimate affine transform." << std::endl;
                    newReference = true;
                    continue;
                }
                

                if (good_new.size() >= 4) {
                    // Estimate an affine (translation + rotation, no scaling) transform using RANSAC.
                    Mat M = estimateAffinePartial2D(filtered_good_ref,
													filtered_good_new,
													noArray(),
													RANSAC,
													0.2,
													4000,
													0.99,
													10);
                    if (!M.empty()) {
                        if (M.type() != CV_32F) {
                            M.convertTo(M, CV_32F);
                        }
                        measured_drift.x = M.at<float>(0, 2);
                        measured_drift.y = M.at<float>(1, 2);
                        validFlow = true;
                    }
                }
            }
            
            // Remove rotation effect from the measured drift.
            if (validFlow) {
                Point3f adjusted = removeRotation(measured_drift);
                float scale_factor = altitude / (float)camera_mat.at<double>(0, 0);
                measured_drift.x = scale_factor * measured_drift.x;
                measured_drift.y = scale_factor * measured_drift.y;
            }
            
            // Update the Kalman filter with the measurement.
            if (validFlow) {
                Mat measurement = (Mat_<float>(2, 1) << measured_drift.x, measured_drift.y);
                kf.correct(measurement);
            }
            Mat prediction = kf.predict();
            Point2f filtered_drift(prediction.at<float>(0), prediction.at<float>(1));
            filtered_drift.x = filtered_drift.x * 0.6 + prevDriftx * 0.4;
            filtered_drift.y = filtered_drift.y * 0.6 + prevDrifty * 0.4;
            // Accumulate drift if needed.
            cumulateDrift(filtered_drift);
            filtered_drift.x += total_drift[0];
            filtered_drift.y += total_drift[1];
            
            // Calculate FPS.
            double fps = 1.0 / elapsed;
            
            // Check for key presses: 'q' to quit, 'c' to capture a new reference frame.

                int key = waitKey(1);

            if (key == 'q') break;
            if (key == 'c') {
                lock_guard<mutex> guard(mtx);
                newReference = true;
                total_drift[0] = 0;
                total_drift[1] = 0;
                cout << "Main frame captured (new reference)!" << endl;
            }
            DriftWriter(filtered_drift.x, filtered_drift.y);
            
            // Display the results.
            showResult(frame, filtered_good_new, fps, &filtered_drift);
        }
        cap.release();
        destroyAllWindows();
    }
};

int main() {
    Drift drift;
    drift.compute();

    const char* sem_name = "/sem_Drift";
    sem_t* semaphore = sem_open(sem_name, 0);
    if (semaphore != SEM_FAILED)
        sem_close(semaphore);

    return 0;
}