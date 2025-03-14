import threading
import time
import numpy as np
import board  # type: ignore
import busio  # type: ignore
import RPi.GPIO as GPIO # type: ignore
import adafruit_vl53l0x  # type: ignore
from pymavlink import mavutil # type: ignore
import posix_ipc  # type: ignore
import mmap

class MavlinkCommunicator:
    def __init__(self, connection_string="/dev/ttyACM0", baudrate=115200):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.master = None
        self.listener_thread = None
        self.running = False
        self.read_lock = threading.Lock() 

    def connect(self):
        """Establish the MAVLink connection and wait for a heartbeat."""
        print(f"Connecting to {self.connection_string} at {self.baudrate} baud...")
        self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate)
        self.master.wait_heartbeat()
        print("Heartbeat received from system:", self.master.target_system)

    def start_listener(self, callback=None):
        """
        Start a background thread to listen for incoming MAVLink messages.
        The optional callback is called with each message.
        """
        self.running = True
        self.listener_thread = threading.Thread(target=self._listener, args=(callback,), daemon=True)
        self.listener_thread.start()

    def _listener(self, callback):
        # Request all data streams at 10Hz
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10, 1
        )

        while self.running:
            try:
                with self.read_lock:
                    msg = self.master.recv_match(blocking=True, timeout=1)
                if msg is not None and callback:
                    callback(msg)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.5)  # Pause briefly before retrying
            time.sleep(0.01)

    def send_command_long(self, command, param1=0, param2=0, param3=0,
                          param4=0, param5=0, param6=0, param7=0):
        if self.master is None:
            print("Not connected!")
            return
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command,
            0,  # Confirmation
            param1, param2, param3, param4, param5, param6, param7
        )
        print(f"Sent COMMAND_LONG: {command}")

    def send_set_mode(self, mode):
        if self.master is None:
            print("Not connected!")
            return
        mode_id = self.master.mode_mapping().get(mode)
        if mode_id is None:
            print(f"[ERROR] Unknown mode: {mode}")
            return
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"Sent set_mode command: {mode}")

    def stop(self):
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        print("Listener stopped.")

class Drone:
    def __init__(self, communicator):
        self.communicator = communicator
        self.master = communicator.master

        self.roll  = 0.0
        self.pitch = 0.0
        self.yaw   = 0.0
        self.alt   = 0.0   # Altitude in meters.
        self.baroAlt = 0.0
        # In guided_nogps mode we use drift in x-y (in meters)
        self.TRIG = 17  # Trigger pin
        self.ECHO = 27
        self.x = 0.0
        self.y = 0.0
        self.gps_initialized = False  # GPS is not used
        self.armed = False
        self.start_time = time.time()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)

    def process_msg(self, msg):
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                self.armed = True
            else:
                self.armed = False
        elif msg_type == "ATTITUDE":
            self.roll, self.pitch, self.yaw = msg.roll, msg.pitch, msg.yaw
        elif msg_type == "GLOBAL_POSITION_INT":
            self.baroAlt = msg.relative_alt / 1000.0
            print(f"msg baroAlt : {self.baroAlt}")
        elif msg_type == "STATUSTEXT":
            severity = msg.severity
            text = msg.text
            if severity <= 3:
                print(f"[ERROR] {text}")
            elif severity <= 5:
                print(f"[WARNING] {text}")
            else:
                print(f"[INFO] {text}")
        elif msg_type == "EKF_STATUS_REPORT":
            print(f"[EKF] Status Flags: {msg.flags}")
        # GPS messages are not processed in guided_nogps mode.

    def connect(self):
        """Start the listener thread using the shared communicator."""
        self.communicator.start_listener(callback=self.process_msg)
        print("Drone connected using MavlinkCommunicator.")

    def set_mode(self, mode):
        self.communicator.send_set_mode(mode)
        print(f"Setting mode to {mode}...")

        with self.communicator.read_lock:
            ack_msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if ack_msg:
            actual_mode = mavutil.mode_string_v10(ack_msg)
            if actual_mode == mode:
                print(f"Mode set to {mode}.")
            else:
                print(f"[ERROR] Failed to set mode to {mode}. Current mode: {actual_mode}")
        else:
            print("[ERROR] No HEARTBEAT received to verify mode change.")

    def arm(self):
        self.communicator.send_command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, param1=1)
        print("Arming drone...")

        timeout = time.time() + 10  
        while not self.armed and time.time() < timeout:
            time.sleep(0.1)
        if self.armed:
            print("Drone armed!")
        else:
            print("Arming failed: Timeout waiting for armed state.")

    def takeoff(self, target_altitude):
        print(f"Taking off to {target_altitude} meters...")
        self.communicator.send_command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, param7=target_altitude)
     
        while self.alt < target_altitude * 0.95:
            print(f"Altitude: {self.alt:.2f} m")
            time.sleep(1)
        print("Reached target altitude!")

    def land(self):
        print("Landing...")
        self.communicator.send_command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)
        
        timeout = time.time() + 10
        while self.armed and time.time() < timeout:
            time.sleep(0.1)
        print("Landed!")

    def AttitudeHandler(self, sem, memory):
        """
        Writes the current attitude (alt, roll, pitch, yaw) to shared memory.
        20 Hz
        """
        while True:
            attitude = f"{self.alt},{self.roll},{self.pitch},{self.yaw}\0".encode("utf-8")
            memory.seek(0)
            memory.write(attitude)
            memory.flush()
            sem.release()
            time.sleep(0.05)

    def UltraBaroHandler(self):

        while True:
            ultrasonic_alt = self.ultrasonic()
            if ultrasonic_alt > 0 and ultrasonic_alt < 3.5:
                self.alt = ultrasonic_alt
            else:
                self.alt = self.baroAlt
            time.sleep(0.05)

    def hold_position_guided_nogps(self, duration, shared_memory, semaphore):
        """
        Holds position in GUIDED_NOGPS mode.
        """
        print("Commanding drone to hold position in GUIDED_NOGPS mode with drift compensation...")
        start_time = time.time()
        while time.time() - start_time < duration:
            # Read drift data (x, y in meters) from shared memory.
            semaphore.acquire()
            try:
                shared_memory.seek(0)
                raw_data = shared_memory.read(1024).decode("utf-8").strip('\x00').strip()
                drift_x, drift_y = 0.0, 0.0
                if raw_data:
                    parts = raw_data.split(',')
                    if len(parts) >= 2:
                        try:
                            drift_x = float(parts[0])
                            drift_y = float(parts[1])
                        except ValueError:
                            print("Error converting drift data to float.")
                    else:
                        print("Error: Incorrect drift format")
                else:
                    print("No drift recieved")
            except Exception as e:
                print(f"Drift exception : {e}")
            finally:
                semaphore.release()

            # Use the current altitude (or a default value) for the z coordinate.
            target_alt = self.alt if self.alt else 1.0
            time_boot_ms = int((time.time() - self.start_time) * 1000)
            # In MAV_FRAME_LOCAL_NED the z value is negative for altitude above ground.
            self.master.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                2032,  # type mask: ignore velocities, accelerations, yaw, etc.
                drift_x, drift_y, -target_alt,  # x, y, z positions
                0, 0, 0,  # velocities
                0, 0, 0,  # accelerations
                0, 0     # yaw, yaw_rate
            )
            time.sleep(0.1)
            
    def ultrasonic(self):
        '''
        Returns ultrasonic distance in meters.
        '''

        # Send a short pulse to trigger the sensor
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)  # 10 µs pulse
        GPIO.output(self.TRIG, False)

        # Wait for the echo pin to go high (with timeout)
        start_time = time.time()
        while GPIO.input(self.ECHO) == 0:
            if time.time() - start_time > 0.05:
                print("[ERROR] Ultrasonic sensor timeout waiting for echo high.")
                return -1  # indicate error

        # Record the time when the echo goes high
        echo_start = time.time()
        # Wait for the echo pin to go low (with timeout)
        while GPIO.input(self.ECHO) == 1:
            if time.time() - echo_start > 0.05:
                print("[ERROR] Ultrasonic sensor timeout waiting for echo low.")
                return -1  # indicate error
        echo_end = time.time()

        # Calculate pulse duration
        pulse_duration = echo_end - echo_start

        # Convert time to distance (Speed of sound = 343 m/s)
        distance = (pulse_duration * 343) / 2

        return round(distance, 2)



if __name__ == "__main__":

    communicator = MavlinkCommunicator("/dev/serial0", 57600)
    communicator.connect()
    drone = Drone(communicator)
    drone.connect()

    # Shared memory for attitude (unchanged)
    sem = posix_ipc.Semaphore("/sem_Attitude", posix_ipc.O_CREAT, initial_value=0)
    shm = posix_ipc.SharedMemory("/shm_Attitude", posix_ipc.O_CREAT, size=1024)
    memory  = mmap.mmap(shm.fd, shm.size)
    shm.close_fd()

    # Shared memory for drift data (x, y in meters)
    drift_sem = posix_ipc.Semaphore("/sem_Drift", posix_ipc.O_CREAT, initial_value=0)
    drift_shm = posix_ipc.SharedMemory("/shm_Drift", posix_ipc.O_CREAT, size=1024)
    drift_memory = mmap.mmap(drift_shm.fd, drift_shm.size)
    drift_shm.close_fd()

    print("Starting threads for attitude...")
    attitude_thread = threading.Thread(target=drone.AttitudeHandler, args=(sem, memory,), daemon=True)
    attitude_thread.start()
    ultraBaro_thread = threading.Thread(target=drone.UltraBaroHandler, daemon=True)
    ultraBaro_thread.start()

    # Set mode to GUIDED_NOGPS (ensure your autopilot supports this mode)
    drone.set_mode("GUIDED_NOGPS")
    drone.arm()
    drone.takeoff(target_altitude=2.0)
    
    # Hold position using drift compensation (reading drift in meters from shared memory)
    drone.hold_position_guided_nogps(duration=180, shared_memory=drift_memory, semaphore=drift_sem)
    
    drone.land()

    print("Mission complete!")
