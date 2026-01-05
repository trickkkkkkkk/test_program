import cv2
import time
import threading
from PySide6.QtCore import QThread, Signal
from gesture_recognizer import MediaPipeGestureRecognizer
from log import debug,error


class VideoCaptureThread(QThread):
    frame_ready = Signal(object)
    detection_status = Signal(dict)  # Emit detection status
    fps_updated = Signal(float)  # Emit FPS updates
    command_detected = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.detecting = True
        self.show_landmarks = True

        # Add exit flag
        self.exiting = False
        self._closed = True  # Track whether resources have been released
        self._lock = threading.RLock()  # Reentrant lock for resource protection

        # Component initialization
        self.gesture = MediaPipeGestureRecognizer()

        # FPS calculation
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.last_command = None
        self.last_face_detected_time = time.time()

    def find_available_camera(self):
        """Automatically detect available camera"""
        debug("Searching for available camera devices...")
        # First try the default cameras (0-9)
        for i in range(10):
            temp_cap = None
            try:
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    ret, frame = temp_cap.read()
                    if ret:
                        temp_cap.release()
                        debug(f"Found available camera at device ID: {i}")
                        return i
            except Exception as e:
                error(f"Error checking camera {i}: {e}")
            finally:
                if temp_cap is not None:
                    try:
                        temp_cap.release()
                    except:
                        error(f"Error temp_cap.release(): {e}")
        error("No available camera device found")
        return None

    def start_capture(self, camera_id=None):
        if camera_id is None:
            camera_id = self.find_available_camera()
            if camera_id is None:
                raise Exception("No available camera device found")

        debug(f"Starting camera capture on device ID: {camera_id}")
        
        # Release existing capture if any
        self._safe_release_capture()
        
        with self._lock:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self._closed = False

        with self._lock:
            self.running = True
            self.frame_count = 0
            self.fps = 0
            self.last_fps_time = time.time()
        self.start()

    def stop_capture(self):
        debug("Stopping camera capture...")
        with self._lock:
            self.running = False
            self.exiting = True

        # Wait for thread to finish, but set timeout
        if self.isRunning():
            self.wait(2000)  # Wait up to 2 seconds

        # Release capture resources
        self._safe_release_capture()

    def _safe_release_capture(self):
        """Safely release camera capture resources with multiple safety checks"""
        try:
            with self._lock:
                if self.cap is not None:
                    try:
                        if not self._closed:
                            debug("Releasing camera capture")
                            self.cap.release()
                    except Exception as e:
                        error(f"Error releasing camera capture: {e}")
                    finally:
                        self.cap = None
                        self._closed = True
                else:
                    # Even if cap is None, mark as closed
                    self._closed = True
        except Exception as e:
            error(f"Error in _safe_release_capture: {e}")
        finally:
            with self._lock:
                self.cap = None
                self._closed = True

    def toggle_detection(self, detecting):
        with self._lock:
            self.detecting = detecting

    def toggle_landmarks(self, show):
        with self._lock:
            self.show_landmarks = show

    def run(self):
        while True:
            # Check exit conditions
            should_continue = False
            with self._lock:
                if (self.running and self.cap is not None and not self._closed):
                    try:
                        should_continue = self.cap.isOpened()
                    except:
                        should_continue = False
                    
            if not should_continue:
                break
                
            try:
                ret, frame = None, None
                cap_valid = False
                with self._lock:
                    cap_valid = self.cap is not None and not self._closed
                    
                if cap_valid:
                    try:
                        ret, frame = self.cap.read()
                    except Exception as e:
                        error(f"Error reading frame: {e}")
                        ret = False
                        
                if ret and frame is not None:
                    # Calculate FPS
                    with self._lock:
                        self.frame_count += 1
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:  # Update once per second
                            self.fps = self.frame_count / (current_time - self.last_fps_time)
                            self.frame_count = 0
                            self.last_fps_time = current_time
                    self.fps_updated.emit(self.fps)

                    processed_frame = frame.copy()
                    detection_result = {}

                    # Process frame if detection is enabled
                    detecting_enabled = False
                    with self._lock:
                        detecting_enabled = self.detecting
                        
                    if detecting_enabled:
                        try:
                            # ---- Gesture recognition ----
                            # 使用封装类逐帧识别手势，并将结果映射为播放/暂停/快进/快退/音量等命令
                            detection_result = {}
                            try:
                                detection_result = self.gesture.process_frame(processed_frame)
                                self.detection_status.emit(detection_result)
                            except Exception as e:
                                error(f"Gesture detection error: {e}")
                                self.detection_status.emit({})

                            command = detection_result.get('cmd', None)
                            # 节流由识别器内部完成，这里只按需发出命令信号
                            if command:
                                debug(f"Command detected: {command}")
                                self.command_detected.emit(command)
                                with self._lock:
                                    self.last_command = command
                            # Draw landmarks (optional)
                            show_landmarks = False
                            with self._lock:
                                show_landmarks = self.show_landmarks
                                
                            # 绘制手势关键点（可选）
                            show_landmarks = False
                            with self._lock:
                                show_landmarks = self.show_landmarks
                            # 为了绘制，需要再次跑一遍 hands.process，这里直接调识别器的 draw 方法
                            #（识别器内部已处理异常）
                            if show_landmarks:
                                try:
                                    import mediapipe as mp
                                    res = mp.solutions.hands.Hands(
                                        static_image_mode=False, max_num_hands=2,
                                        min_detection_confidence=0.6, min_tracking_confidence=0.6
                                    ).process(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                                    self.gesture.draw_landmarks(processed_frame, res)
                                except Exception as e:
                                    error(f"draw landmarks err: {e}")

                            # Emit command signal
                            if command and command != self.last_command:
                                debug(f"Command detected: {command}")
                                self.command_detected.emit(command)
                                with self._lock:
                                    self.last_command = command

                        except Exception as e:
                            error(f"Detection error: {e}")
                            # Emit empty status to indicate detection failure
                            self.detection_status.emit({})
                    else:
                        # If detection is disabled, emit empty status
                        self.detection_status.emit({})

                    # Emit frame ready signal
                    self.frame_ready.emit(processed_frame)

                    time.sleep(0.03)  # ~30 FPS
                else:
                    # If we can't read a frame, stop the capture
                    error("Cannot read frame from camera, stopping capture")
                    break
            except Exception as e:
                error(f"Error in camera capture loop: {e}")
                break

        # Release resources when thread exits
        self._safe_release_capture()
        self.finished.emit()

    def __del__(self):
        """Ensure resources are released when object is destroyed"""
        try:
            self._safe_release_capture()
        except Exception as e :
            error(f"Error release resources: {e}")