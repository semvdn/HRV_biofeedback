import cv2
import numpy as np
import time
from collections import deque
import math
from heart_rate_monitor import HeartRateMonitor

# -----------------------------------------------------------------------------
# 1. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def interpolate_color(color_start, color_end, t):
    """Linearly interpolates between two BGR colors.

    Args:
        color_start (tuple): The starting BGR color tuple.
        color_end (tuple): The ending BGR color tuple.
        t (float): The interpolation factor, ranging from 0.0 to 1.0.

    Returns:
        tuple: The interpolated BGR color tuple.
    """
    b = int(color_start[0] * (1 - t) + color_end[0] * t)
    g = int(color_start[1] * (1 - t) + color_end[1] * t)
    r = int(color_start[2] * (1 - t) + color_end[2] * t)
    return (b, g, r)

def draw_gradient_rect(img, pt1, pt2, color_top, color_bottom):
    x1, y1 = pt1
    x2, y2 = pt2
    height = y2 - y1
    width = x2 - x1 
    if width <= 0 or height <= 0: return
    gradient = np.zeros((height, 1, 3), dtype=np.uint8)
    for i in range(height):
        ratio = i / height
        gradient[i] = interpolate_color(color_top, color_bottom, ratio)
    gradient = cv2.resize(gradient, (width, height))
    img[y1:y2, x1:x2] = gradient

# -----------------------------------------------------------------------------
# 2. TRACKING ENGINE
# -----------------------------------------------------------------------------

class BreathingTracker:
    """Tracks breathing rate by detecting chest movements using optical flow."""
    def __init__(self, cascade_path=None):
        self.feature_params = dict(maxCorners=80, qualityLevel=0.15, minDistance=8, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_file)
        if self.cascade.empty():
            raise IOError(f"Failed to load Haar Cascade: {cascade_file}")

        self.old_gray = None
        self.p0 = None
        self.roi_box = None
        self.tracking_active = False
        
        self.signal_buffer = deque(maxlen=300) 
        self.position_buffer = deque(maxlen=300) 
        self.current_position = 0
        self.smoothed_signal = 0
        self.velocity_buffer = deque(maxlen=5) 
        self.current_direction = 0 

        self.breathing_rate_bpm = 0.0
        self.last_inhale_times = deque(maxlen=10) 

    def reset(self):
        """Resets the tracker to its initial state."""
        self.tracking_active = False
        self.p0 = None
        self.signal_buffer.clear()
        self.position_buffer.clear()
        self.current_position = 0
        self.breathing_rate_bpm = 0.0
        self.last_inhale_times.clear()

    def detect_and_init(self, frame):
        """Detects a face and initializes the tracker on the chest/shoulder region.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            bool: True if tracking is successfully initialized, False otherwise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            roi_x, roi_y = x - int(w * 0.5), y + int(h * 1.0)
            roi_w, roi_h = int(w * 2.0), int(h * 1.5)
            H, W = frame.shape[:2]
            roi_x, roi_y = max(0, roi_x), max(0, roi_y)
            roi_w, roi_h = min(W - roi_x, roi_w), min(H - roi_y, roi_h)
            self.roi_box = (roi_x, roi_y, roi_w, roi_h)
            roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            self.p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **self.feature_params)
            if self.p0 is not None:
                for i in range(len(self.p0)):
                    self.p0[i][0][0] += roi_x
                    self.p0[i][0][1] += roi_y
                self.old_gray = gray.copy()
                self.tracking_active = True
                return True
        return False

    def update(self, frame):
        """Updates the tracker with a new frame.

        Args:
            frame (np.ndarray): The new video frame.

        Returns:
            tuple: A tuple containing a boolean indicating if tracking is active,
                   and the processed frame.
        """
        if not self.tracking_active or self.p0 is None:
            return self.detect_and_init(frame), frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is not None and st is not None:
            good_new, good_old = p1[st==1], self.p0[st==1]
            if len(good_new) < 10:
                self.tracking_active = False
                return False, frame
            dys = [- (new[1] - old[1]) for new, old in zip(good_new, good_old)]
            if dys:
                avg_dy = np.mean(dys)
                amplified_val = avg_dy * 8.0 
                self.smoothed_signal = (self.smoothed_signal * 0.9) + (amplified_val * 0.1)
                self.signal_buffer.append(self.smoothed_signal)
                self.current_position += self.smoothed_signal
                self.current_position *= 0.99
                self.position_buffer.append(self.current_position)
                self.velocity_buffer.append(self.smoothed_signal)
                avg_vel = np.mean(self.velocity_buffer)
                if avg_vel > 0.1: self.current_direction = 1
                elif avg_vel < -0.1: self.current_direction = -1
                else: self.current_direction = 0
            
            if len(self.position_buffer) > 1:
                if self.position_buffer[-2] < 0 and self.position_buffer[-1] >= 0:
                    self.last_inhale_times.append(time.time())

            if len(self.last_inhale_times) > 1:
                durations = np.diff(self.last_inhale_times)
                avg_duration = np.mean(durations)
                if avg_duration > 0:
                    self.breathing_rate_bpm = 60.0 / avg_duration
            
            if self.last_inhale_times and (time.time() - self.last_inhale_times[-1] > 15):
                self.breathing_rate_bpm = 0.0

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.tracking_active = False
        return self.tracking_active, frame

    def draw_debug(self, frame):
        """Draws debugging information on the frame.

        Args:
            frame (np.ndarray): The frame to draw on.

        Returns:
            np.ndarray: The frame with debugging information.
        """
        if self.roi_box:
            x, y, w, h = self.roi_box
            color = (255, 255, 0)
            l, t = 20, 2
            cv2.line(frame, (x, y), (x+l, y), color, t); cv2.line(frame, (x, y), (x, y+l), color, t)
            cv2.line(frame, (x+w, y), (x+w-l, y), color, t); cv2.line(frame, (x+w, y), (x+w, y+l), color, t)
            cv2.line(frame, (x, y+h), (x+l, y+h), color, t); cv2.line(frame, (x, y+h), (x, y+h-l), color, t)
            cv2.line(frame, (x+w, y+h), (x+w-l, y+h), color, t); cv2.line(frame, (x+w, y+h), (x+w, y+h-l), color, t)
            status_text = "TRACKING ACTIVE" if self.tracking_active else "SEARCHING..."
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if self.p0 is not None and self.tracking_active:
            pt_color = (0, 255, 0) if self.current_direction == 1 else ((0, 0, 255) if self.current_direction == -1 else (0, 255, 255))
            for point in self.p0:
                a, b = point.ravel()
                cv2.circle(frame, (int(a), int(b)), 2, pt_color, -1)
        return frame

# -----------------------------------------------------------------------------
# 3. APPLICATION LOGIC
# -----------------------------------------------------------------------------

class App:
    """Main application class for the biofeedback tool."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        self.width, self.height = 1280, 720
        self.window_name = "Biofeedback App"
        
        self.tracker = BreathingTracker()
        self.hr_monitor = HeartRateMonitor()
        
        self.target_bpm = 6.0
        self.inhale_ratio = 0.4
        self.start_time = time.time()

        # Resonance finder (automatic sweep of candidate pacer rates)
        self.resonance_mode = False
        self.resonance_frequencies = [6.5, 6.0, 5.5, 5.0, 4.5]
        self.resonance_step_duration = 90.0   # seconds at each pace
        self.resonance_warmup = 30.0          # initial seconds ignored for averaging
        self.resonance_index = 0
        self.resonance_step_start = None
        self.resonance_stats = {
            f: {"sum_hrv": 0.0, "count": 0} for f in self.resonance_frequencies
        }
        self.resonance_result_freq = None
        self.resonance_result_hrv = 0.0
        self.pending_resonance_action = None
        
        # Smooth pacer transitions
        self.current_bpm = self.target_bpm
        self.transition_start_bpm = self.target_bpm
        self.transition_start_time = None
        
        self.show_camera = True
        self.sync_score = 50.0
        
        self.c_bg_top = (60, 60, 60)
        self.c_bg_bot = (10, 10, 10)
        
        # Pacer Colors (Toned down inhale)
        self.c_exhale = (255, 100, 50)
        self.c_inhale = (50, 190, 240)
        
        # Text Colors
        self.c_text_exhale = (210, 210, 210)
        self.c_text_inhale = (20, 20, 20)
        self.c_text_main = (200, 200, 200)
        self.c_text_dim = (150, 150, 150)
        self.c_box_bg = (50, 50, 50)
        self.c_box_border = (120, 120, 120)

    def reset_resonance_finder(self):
        """Resets the resonance finder to its initial state."""
        self.resonance_mode = False
        self.resonance_index = 0
        self.resonance_step_start = None
        self.resonance_stats = {
            f: {"sum_hrv": 0.0, "count": 0} for f in self.resonance_frequencies
        }
        self.resonance_result_freq = None
        self.resonance_result_hrv = 0.0
        self.pending_resonance_action = None

    def start_resonance_finder(self):
        """Queues the activation of the resonance finder."""
        if self.resonance_mode:
            return
        self.pending_resonance_action = {"type": "start"}

    def stop_resonance_finder(self):
        """Stops the resonance finder."""
        self.resonance_mode = False
        self.resonance_step_start = None
        self.pending_resonance_action = None

    def update_resonance_finder(self, hrv_rmssd):
        """Updates the resonance finder with the latest IBI-based HRV.
        
        Args:
            hrv_rmssd (float): The current HRV (RMSSD) in milliseconds.
        """
        if not self.resonance_mode or not self.resonance_frequencies:
            return

        now = time.time()
        if self.resonance_step_start is None:
            self.resonance_step_start = now

        current_freq = self.resonance_frequencies[self.resonance_index]
        elapsed = now - self.resonance_step_start

        # Only accumulate HRV once it has had time to stabilize,
        # and only within the nominal step duration.
        if elapsed >= self.resonance_warmup and elapsed <= self.resonance_step_duration and hrv_rmssd > 0:
            stats = self.resonance_stats[current_freq]
            stats["sum_hrv"] += hrv_rmssd
            stats["count"] += 1

        # When the step time is up, schedule a change in breathing rate,
        # but only actually apply it at the next apex or bottom of the pacer.
        if elapsed >= self.resonance_step_duration and self.pending_resonance_action is None:
            next_index = self.resonance_index + 1
            if next_index >= len(self.resonance_frequencies):
                # Finished sweep: pick best frequency based on average HRV
                best_freq, best_hrv = self.get_best_resonance_frequency()
                if best_freq is not None:
                    self.resonance_result_freq = best_freq
                    self.resonance_result_hrv = best_hrv
                # Defer switching the pacer to the best frequency until an extreme
                self.pending_resonance_action = {
                    "type": "finish",
                    "freq": self.resonance_result_freq,
                }
            else:
                # Queue transition to the next test frequency
                self.pending_resonance_action = {
                    "type": "next_step",
                    "freq": self.resonance_frequencies[next_index],
                    "index": next_index,
                }

    def maybe_apply_pending_resonance_change(self, progress):
        """Applies any pending resonance changes at pacer extremes.

        Args:
            progress (float): The current progress of the pacer cycle.
        """
        if not self.pending_resonance_action:
            return

        # Extremes of the pacer motion are where progress ~0 (bottom) or ~1 (apex).
        if 0.05 < progress < 0.95:
            return

        action = self.pending_resonance_action
        self.pending_resonance_action = None
        now = time.time()

        if action["type"] == "start":
            self.reset_resonance_finder()
            self.resonance_mode = True
            self.resonance_index = 0
            if self.resonance_frequencies:
                self.set_target_bpm(self.resonance_frequencies[self.resonance_index])
            self.resonance_step_start = now
        elif action["type"] == "next_step":
            if not self.resonance_mode: return
            self.resonance_index = action["index"]
            self.set_target_bpm(action["freq"])
            self.resonance_step_start = now
        elif action["type"] == "finish":
            if not self.resonance_mode: return
            freq = action.get("freq")
            if freq is not None:
                self.set_target_bpm(freq)
            self.stop_resonance_finder()

    def set_target_bpm(self, new_bpm):
        if self.target_bpm == new_bpm: return
        self.transition_start_time = time.time()
        self.transition_start_bpm = self.current_bpm
        self.target_bpm = new_bpm

    def get_best_resonance_frequency(self):
        """Determines the best resonance frequency based on collected stats.

        Returns:
            tuple: A tuple containing the best frequency and the corresponding HRV.
        """
        best_freq = None
        best_hrv = 0.0
        for freq, stats in self.resonance_stats.items():
            if stats["count"] > 0:
                avg = stats["sum_hrv"] / stats["count"]
                if avg > best_hrv:
                    best_hrv = avg
                    best_freq = freq
        return best_freq, best_hrv
        
    def get_pacer_state(self):
        """Calculates the current state of the breathing pacer.

        Returns:
            tuple: A tuple containing the pacer radius, color, phase type, and progress.
        """
        now = time.time()
        transition_duration = 3.0
        if self.transition_start_time and now < self.transition_start_time + transition_duration:
            t = (now - self.transition_start_time) / transition_duration
            ease_t = 0.5 - 0.5 * math.cos(t * math.pi)
            self.current_bpm = self.transition_start_bpm + (self.target_bpm - self.transition_start_bpm) * ease_t
        else:
            self.current_bpm = self.target_bpm
            self.transition_start_time = None

        cycle_dur = 60.0 / self.current_bpm
        elapsed = now - self.start_time
        local_t = (elapsed % cycle_dur) / cycle_dur
        if local_t < self.inhale_ratio:
            norm_t = local_t / self.inhale_ratio
            phase_type, progress = 1, 0.5 - 0.5 * math.cos(norm_t * math.pi)
        else:
            norm_t = (local_t - self.inhale_ratio) / (1.0 - self.inhale_ratio)
            phase_type, progress = -1, 0.5 + 0.5 * math.cos(norm_t * math.pi)
        
        current_color = interpolate_color(self.c_exhale, self.c_inhale, progress)
        min_r, max_r = 60, 160
        radius = int(min_r + (max_r - min_r) * progress)
        return radius, current_color, phase_type, progress

    # ########################################################################
    # OPTIMIZED FLUID RENDERER (Slice-Based Calculation)
    # ########################################################################
    def add_blob_optimized(self, density_map, x, y, r, weight):
        """Adds a blob to the density map using localized slicing (Performance Optimization)."""
        sigma = r / 1.8
        # Limit calculation to 3 sigma (covers >99% of influence)
        limit = int(3 * sigma)
        
        h, w = density_map.shape
        y_min = max(0, y - limit)
        y_max = min(h, y + limit)
        x_min = max(0, x - limit)
        x_max = min(w, x + limit)
        
        if x_min >= x_max or y_min >= y_max: return

        # Create small local grid
        local_y, local_x = np.ogrid[y_min-y:y_max-y, x_min-x:x_max-x]
        dist_sq = local_x**2 + local_y**2
        blob = np.exp(-dist_sq / (2 * sigma**2)) * weight
        
        density_map[y_min:y_max, x_min:x_max] += blob

    def draw_fluid_pacer(self, ui, cx, cy, radius, color, phase_dir, progress):
        """Draws the fluid pacer animation on the UI.

        Args:
            ui (np.ndarray): The UI frame to draw on.
            cx (int): The center x-coordinate.
            cy (int): The center y-coordinate.
            radius (int): The current radius of the pacer.
            color (tuple): The current color of the pacer.
            phase_dir (int): The current phase direction (inhale/exhale).
            progress (float): The current progress of the pacer cycle.
        """
        # ROI setup
        box_size = int(radius * 2.5) + 60
        top_left_x = max(0, cx - box_size // 2)
        top_left_y = max(0, cy - box_size // 2)
        
        roi = ui[top_left_y : top_left_y + box_size, top_left_x : top_left_x + box_size]
        if roi.shape[0] == 0 or roi.shape[1] == 0: return
        
        h, w = roi.shape[:2]
        local_cx, local_cy = w // 2, h // 2

        # 1. Physics (Adjusted for "Center Attraction")
        t = time.time()
        # Main body breathing
        organic_r = radius + math.sin(t * 1.5) * 1.5 

        # Blob 1: Slow circular orbit with variable radius (breathing in and out of center)
        # distance oscillates between 0.14*r and 0.30*r
        angle1 = t * 0.8
        dist1 = radius * (0.22 + 0.08 * math.sin(t * 0.43)) 
        b1_x = int(local_cx + math.cos(angle1) * dist1)
        b1_y = int(local_cy + math.sin(angle1) * dist1)
        b1_r = int(radius * 0.6) # Slightly larger to melt better near center
        
        # Blob 2: Vertical drifting with center pull
        angle2 = t * 1.1 + 2.0
        y_dist2 = radius * (0.25 + 0.1 * math.cos(t * 0.6)) # Oscillates vertical distance
        b2_x = int(local_cx + math.sin(t * 0.5) * (radius * 0.1))
        b2_y = int(local_cy + math.sin(angle2) * y_dist2) 
        b2_r = int(radius * 0.5)

        # Blob 3: Counter-orbit, tighter to core
        angle3 = t * -0.9 + 4.0
        dist3 = radius * (0.18 + 0.05 * math.sin(t * 1.1)) # Very tight orbit
        b3_x = int(local_cx + math.cos(angle3) * dist3)
        b3_y = int(local_cy + math.sin(angle3) * dist3)
        b3_r = int(radius * 0.55)

        # 2. Render Density Map (Optimized with Slicing)
        mask_canvas = np.zeros((h, w), dtype=np.float32)
        
        self.add_blob_optimized(mask_canvas, local_cx, local_cy, organic_r, 1.45)
        self.add_blob_optimized(mask_canvas, b1_x, b1_y, b1_r, 0.7)
        self.add_blob_optimized(mask_canvas, b2_x, b2_y, b2_r, 0.7)
        self.add_blob_optimized(mask_canvas, b3_x, b3_y, b3_r, 0.7)
        
        # 3. Soft Thresholding
        alpha_mask = np.clip((mask_canvas - 0.5) / 0.1, 0, 1)
        alpha_3c = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

        # 4. Create Color Layer
        color_layer = np.zeros((h, w, 3), dtype=np.uint8)
        color_layer[:] = color 

        # Internal Highlights (Toned down)
        highlight_color = interpolate_color(color, (255, 255, 255), 0.25)
        off = int(radius * 0.15)
        cv2.circle(color_layer, (b1_x - off, b1_y - off), int(b1_r * 0.4), highlight_color, -1, cv2.LINE_AA)
        cv2.circle(color_layer, (b2_x - off, b2_y - off), int(b2_r * 0.4), highlight_color, -1, cv2.LINE_AA)
        
        # Optimization: Cap the Blur Kernel size
        blur_k = min(int(radius * 0.8) | 1, 41) 
        color_layer = cv2.GaussianBlur(color_layer, (blur_k, blur_k), 0)

        # Surface Specular Gloss
        specular_mask = np.clip((mask_canvas - 0.85) / 0.1, 0, 1)
        specular_mask = cv2.merge([specular_mask, specular_mask, specular_mask])
        
        # 5. Final Composition
        roi_float = roi.astype(float)
        color_float = color_layer.astype(float)
        
        blended = color_float * alpha_3c + roi_float * (1.0 - alpha_3c)
        
        # (Glint removed here)

        # 6. Outer Glow (Reduced intensity)
        glow_intensity = np.clip((mask_canvas * 0.15), 0, 0.20)
        glow_3c = cv2.merge([glow_intensity, glow_intensity, glow_intensity])
        glow_color = np.zeros_like(roi_float)
        glow_color[:] = color
        
        final_result = blended + (glow_color * glow_3c * (1.0 - alpha_3c))
        final_result = np.clip(final_result, 0, 255).astype(np.uint8)
        
        ui[top_left_y : top_left_y + h, top_left_x : top_left_x + w] = final_result

    def draw_dashboard(self, frame, width, height):
        """Draws the main dashboard UI.

        Args:
            frame (np.ndarray): The current video frame.
            width (int): The width of the UI.
            height (int): The height of the UI.

        Returns:
            np.ndarray: The rendered UI frame.
        """
        self.width, self.height = width, height
        ui = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        draw_gradient_rect(ui, (0,0), (self.width, self.height), self.c_bg_top, self.c_bg_bot)

        pacer_state = self.get_pacer_state()
        pacer_radius, pacer_color, phase_dir, progress = pacer_state
        # Apply any queued resonance step change only when the pacer is at
        # the top or bottom of its cycle so the speed change is unobtrusive.
        self.maybe_apply_pending_resonance_change(progress)
        cx, cy = self.width // 2, self.height // 2 - int(self.height * 0.1)
        min_r_rel, max_r_rel = 0.08 * self.height, 0.22 * self.height
        radius = int(min_r_rel + (max_r_rel - min_r_rel) * progress)

        # Guides
        cv2.circle(ui, (cx, cy), int(min_r_rel), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.circle(ui, (cx, cy), int(max_r_rel), (80, 80, 80), 1, cv2.LINE_AA)
        
        # Draw Fluid Pacer
        # Compensate for fluid edge softness by multiplying radius by 1.15
        self.draw_fluid_pacer(ui, cx, cy, int(radius * 1.15), pacer_color, phase_dir, progress)
        
        # Text
        txt = "INHALE" if phase_dir == 1 else "EXHALE"
        font_scale = self.height / 900.0
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_col = interpolate_color(self.c_text_exhale, self.c_text_inhale, progress)
        cv2.putText(ui, txt, (cx - ts[0]//2, cy + ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, 2)

        match = False
        if self.tracker.tracking_active:
            if phase_dir == self.tracker.current_direction:
                self.sync_score += 0.5; match = True
            else: self.sync_score -= 0.2
        self.sync_score = max(0, min(100, self.sync_score))
        bar_w, bar_h = int(self.width * 0.3), int(self.height * 0.02)
        bar_x, bar_y = cx - bar_w//2, cy - int(self.height * 0.30)
        cv2.rectangle(ui, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.c_box_bg, -1)
        fill_w = int((self.sync_score / 100) * bar_w)
        bar_col = (0, 255, 0) if match else (120, 120, 120)
        cv2.rectangle(ui, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.putText(ui, "RHYTHM MATCH", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c_text_dim, 1)

        if self.show_camera:
            cam_w, cam_h = int(self.width * 0.25), int(self.width * 0.25 * 0.75)
            debug_frame = self.tracker.draw_debug(frame.copy())
            debug_frame = self.hr_monitor.draw_debug(debug_frame)
            thumb = cv2.resize(debug_frame, (cam_w, cam_h))
            margin = int(self.width * 0.02)
            y_offset, x_offset = margin, self.width - cam_w - margin
            cv2.rectangle(ui, (x_offset-2, y_offset-2), (x_offset+cam_w+2, y_offset+cam_h+2), (100,100,100), 1)
            ui[y_offset:y_offset+cam_h, x_offset:x_offset+cam_w] = thumb
            font_scale_small = self.height / 1500.0
            cv2.putText(ui, "Green: Inhale (Up)", (x_offset, y_offset + cam_h + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0,255,0), 1)
            cv2.putText(ui, "Red: Exhale (Down)", (x_offset, y_offset + cam_h + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0,0,255), 1)

        graph_w, graph_h = int(self.width * 0.5), int(self.height * 0.15)
        g_x = (self.width - graph_w) // 2
        g_y = self.height - graph_h - int(self.height * 0.05)
        center_y = g_y + graph_h // 2
        cv2.line(ui, (g_x, center_y), (g_x + graph_w, center_y), (60, 60, 60), 1)
        pts = list(self.tracker.position_buffer)
        if len(pts) > 1:
            smoothed_pts = self.smooth_data(pts, window_size=5)
            max_abs_val = max(abs(p) for p in smoothed_pts) if smoothed_pts else 1.0
            if max_abs_val < 1.0: max_abs_val = 1.0
            view_range = max_abs_val * 2.5
            pixel_pts = []
            for i, y_val in enumerate(smoothed_pts):
                py = int(center_y - (y_val / view_range) * graph_h)
                px = int(g_x + i * (graph_w / len(smoothed_pts)))
                pixel_pts.append((px, py))
            cv2.polylines(ui, [np.array(pixel_pts, dtype=np.int32)], isClosed=False, color=(255, 255, 220), thickness=2, lineType=cv2.LINE_AA)
            fill_poly = np.array(pixel_pts + [(g_x + graph_w, center_y), (g_x, center_y)], dtype=np.int32)
            overlay = ui.copy(); cv2.fillPoly(overlay, [fill_poly], self.c_inhale, cv2.LINE_AA); ui = cv2.addWeighted(ui, 1.0, overlay, 0.3, 0)

            breathing_bpm = self.tracker.breathing_rate_bpm
            if breathing_bpm > 0 and pixel_pts:
                last_pt = pixel_pts[-1]
                bpm_text = f"{breathing_bpm:.1f}"
                font_scale, text_color = 0.6, (255, 255, 220)
                (text_w, text_h), _ = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_x, text_y = last_pt[0] + 15, last_pt[1] + text_h // 2
                cv2.putText(ui, bpm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(ui, bpm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

        hr_graph_w, hr_graph_h = int(self.width * 0.5), int(self.height * 0.12)
        hr_g_x = (self.width - hr_graph_w) // 2
        hr_g_y = g_y - hr_graph_h - int(self.height * 0.02)
        hr_source, hr_bpm, hr_pts, hrv_pts = (
            self.hr_monitor.get_active_source(),
            self.hr_monitor.get_bpm(),
            list(self.hr_monitor.bpm_history),
            list(self.hr_monitor.hrv_history),
        )

        # IBI-based HRV (RMSSD, ms) when Bluetooth is active
        hrv_rmssd_ms = 0.0
        if hr_source == "Bluetooth":
            hrv_rmssd_ms = self.hr_monitor.get_hrv_rmssd_ms(window_sec=60.0)

        # Feed IBI-based HRV into resonance finder (if active)
        self.update_resonance_finder(hrv_rmssd_ms)

        # Dynamic Scaling
        all_values = [p for p in hr_pts if p > 30]
        
        if all_values:
            y_min_val = min(all_values)
            y_max_val = max(all_values)
            pad = (y_max_val - y_min_val) * 0.15
            if pad < 5: pad = 5
            view_min = y_min_val - pad
            view_max = y_max_val + pad
        else:
            view_min, view_max = 50, 120
        
        if view_max - view_min < 20:
            mid = (view_max + view_min) / 2
            view_min = mid - 10
            view_max = mid + 10

        # Create a separate vertical axis for HRV on the right
        hrv_view_min, hrv_view_max = 0, 150  # Typical RMSSD range
        if any(hrv_pts):
            min_hrv = min(v for v in hrv_pts if v > 0)
            max_hrv = max(hrv_pts)
            hrv_pad = (max_hrv - min_hrv) * 0.15
            if hrv_pad < 10:
                hrv_pad = 10
            hrv_view_min = max(0, min_hrv - hrv_pad)
            hrv_view_max = max_hrv + hrv_pad

        if len(hr_pts) > 1:
            hr_pixel_pts = []
            for i, bpm_val in enumerate(hr_pts):
                clamped_bpm = max(view_min, min(view_max, bpm_val))
                y_ratio = (clamped_bpm - view_min) / (view_max - view_min)
                py = int(hr_g_y + hr_graph_h - (y_ratio * hr_graph_h))
                px = int(hr_g_x + i * (hr_graph_w / len(hr_pts)))
                hr_pixel_pts.append((px, py))
            
            # Draw the HRV line using the right-hand scale
            if len(hrv_pts) > 1:
                hrv_pixel_pts = []
                for i, hrv_val in enumerate(hrv_pts):
                    clamped_hrv = max(hrv_view_min, min(hrv_view_max, hrv_val))
                    y_ratio = (clamped_hrv - hrv_view_min) / (hrv_view_max - hrv_view_min)
                    py = int(hr_g_y + hr_graph_h - (y_ratio * hr_graph_h))
                    px = int(hr_g_x + i * (hr_graph_w / len(hrv_pts)))
                    hrv_pixel_pts.append((px, py))
                cv2.polylines(ui, [np.array(hrv_pixel_pts, dtype=np.int32)], isClosed=False, color=(255, 200, 100), thickness=2, lineType=cv2.LINE_AA)
            
            # Draw Right-side (HRV) Axis Labels
            font_tiny = 0.4
            cv2.putText(ui, f"{int(hrv_view_max)}ms", (hr_g_x + hr_graph_w + 5, hr_g_y + 10), cv2.FONT_HERSHEY_SIMPLEX, font_tiny, (200, 200, 200), 1)
            cv2.putText(ui, f"{int(hrv_view_min)}", (hr_g_x + hr_graph_w + 5, hr_g_y + hr_graph_h), cv2.FONT_HERSHEY_SIMPLEX, font_tiny, (200, 200, 200), 1)

            cv2.polylines(ui, [np.array(hr_pixel_pts, dtype=np.int32)], isClosed=False, color=(50, 50, 255), thickness=2, lineType=cv2.LINE_AA)
            
            if hr_pixel_pts:
                last_pt = hr_pixel_pts[-1]
                base_radius = 5
                if hr_source == 'Bluetooth' and hr_bpm > 0:
                    pulse_rate = hr_bpm / 60.0
                    pulse_effect = (math.sin(time.time() * pulse_rate * 2 * math.pi) + 1) / 2
                    pulse_radius_offset = int(pulse_effect * 4)
                    cv2.circle(ui, last_pt, base_radius + pulse_radius_offset, (100, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(ui, last_pt, base_radius + pulse_radius_offset + 3, (100, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(ui, last_pt, base_radius, (100, 100, 255), -1, cv2.LINE_AA)
                    cv2.circle(ui, last_pt, base_radius + 3, (100, 100, 255), 1, cv2.LINE_AA)
                if hr_bpm > 0:
                    bpm_text = f"{int(hr_bpm)}"
                    font_scale, text_color = 0.7, (255, 255, 255)
                    (text_w, text_h), _ = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    text_x, text_y = last_pt[0] - 50, last_pt[1] + text_h // 2
                    cv2.putText(ui, bpm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(ui, bpm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
        
        margin = int(self.width * 0.02)
        font_scale_title, font_scale_box_head, font_scale_box_val = self.height / 700.0, self.height / 1300.0, self.height / 800.0
        cv2.putText(ui, "HRV BIOFEEDBACK TRAINER", (margin, int(self.height * 0.06)), cv2.FONT_HERSHEY_DUPLEX, font_scale_title * 0.8, self.c_text_main, 1)

        # Resonance finder status
        status_text = None
        if self.resonance_mode and self.resonance_frequencies:
            current_freq = self.resonance_frequencies[self.resonance_index]
            remaining = 0
            if self.resonance_step_start is not None:
                elapsed = time.time() - self.resonance_step_start
                remaining = max(0, int(self.resonance_step_duration - elapsed))
            status_text = f"Resonance finder: {current_freq:.1f} BPM ({remaining}s in this step)"
        elif self.resonance_result_freq is not None and self.resonance_result_hrv > 0:
            status_text = f"Suggested resonance: {self.resonance_result_freq:.1f} BPM (avg HRV {self.resonance_result_hrv:.0f}ms)"
        if status_text:
            cv2.putText(ui, status_text, (margin, int(self.height * 0.09)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_main, 1)

        box_w, box_h = int(self.width * 0.18), int(self.height * 0.1)
        y_start = int(self.height * 0.1)
        
        # 1. PACER RATE
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_bg, -1); cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_border, 1)
        cv2.putText(ui, "PACER RATE", (margin + 15, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_dim, 1)
        cv2.putText(ui, f"{self.target_bpm:.1f} BPM", (margin + 15, y_start + 65), cv2.FONT_HERSHEY_DUPLEX, font_scale_box_val, self.c_text_main, 2)
        
        # 2. RATIO
        y_start += box_h + 10
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_bg, -1); cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_border, 1)
        cv2.putText(ui, "INHALE/EXHALE RATIO", (margin + 15, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_dim, 1)
        cv2.putText(ui, f"{self.inhale_ratio*100:.0f}% / {(1-self.inhale_ratio)*100:.0f}%", (margin + 15, y_start + 65), cv2.FONT_HERSHEY_DUPLEX, font_scale_box_val*0.8, self.c_text_main, 2)

        # 3. BREATHING RATE
        y_start += box_h + 10
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_bg, -1); cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_border, 1)
        cv2.putText(ui, "BREATHING RATE", (margin + 15, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_dim, 1)
        breathing_bpm = self.tracker.breathing_rate_bpm
        breathing_text = f"{breathing_bpm:.1f} BPM" if breathing_bpm > 0 else "---"
        cv2.putText(ui, breathing_text, (margin + 15, y_start + 65), cv2.FONT_HERSHEY_DUPLEX, font_scale_box_val, self.c_text_main, 2)

        # 4. HEART RATE
        y_start += box_h + 10
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_bg, -1); cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_border, 1)
        hr_source_text = f"HEART RATE ({hr_source})"
        cv2.putText(ui, hr_source_text, (margin + 15, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_dim, 1)
        hr_text = f"{hr_bpm:.1f} BPM" if hr_bpm > 0 else "---"
        cv2.putText(ui, hr_text, (margin + 15, y_start + 65), cv2.FONT_HERSHEY_DUPLEX, font_scale_box_val, self.c_text_main, 2)

        # 5. HRV display: use IBI-based RMSSD
        y_start += box_h + 10
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_bg, -1)
        cv2.rectangle(ui, (margin, y_start), (margin + box_w, y_start + box_h), self.c_box_border, 1)

        label = "HRV (RMSSD, IBI)"
        value_text = "---"
        if hr_source == "Bluetooth" and hrv_rmssd_ms > 0:
            value_text = f"{hrv_rmssd_ms:.0f} ms"

        cv2.putText(ui, label, (margin + 15, y_start + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_box_head, self.c_text_dim, 1)
        cv2.putText(ui, value_text, (margin + 15, y_start + 65),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale_box_val, self.c_text_main, 2)

        instr_x, y_start, line_height = margin, self.height - int(self.height * 0.25), int(self.height * 0.03)
        controls = ["CONTROLS:", "[UP/W]: Faster", "[DOWN/S]: Slower", "[D]: Inhale Ratio +", "[A]: Inhale Ratio -", "[F]: Resonance Finder Sweep", "[R]: Reset Tracker", "[C]: Toggle Camera", "[Q]: Quit"]
        for i, text in enumerate(controls):
            font = cv2.FONT_HERSHEY_DUPLEX if i == 0 else cv2.FONT_HERSHEY_SIMPLEX
            color = self.c_text_dim if i == 0 else self.c_text_main
            cv2.putText(ui, text, (instr_x, y_start), font, font_scale_box_head, color, 1)
            y_start += line_height
        
        return ui

    def smooth_data(self, data, window_size=5):
        """Smooths data using a simple moving average.

        Args:
            data (list): The data to smooth.
            window_size (int): The size of the moving average window.

        Returns:
            list: The smoothed data.
        """
        if not data or window_size < 2: return data
        padded = [data[0]] * (window_size - 1) + list(data)
        return [sum(padded[i : i + window_size]) / window_size for i in range(len(data))]

    def run(self):
        """Main application loop."""
        print("Starting App...")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        try:
            while True:
                # Exit if the window is closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1: break
                
                # Handle window resizing
                _, _, w, h = cv2.getWindowImageRect(self.window_name)
                if w > 0 and h > 0: self.width, self.height = w, h

                # Read a frame from the webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break
                frame = cv2.flip(frame, 1)
                
                # Update trackers
                self.tracker.update(frame)
                self.hr_monitor.update()
                
                # Render the UI
                ui = self.draw_dashboard(frame, self.width, self.height)
                cv2.imshow(self.window_name, ui)
                
                # Handle user input
                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'): break
                elif key == ord('r'): self.tracker.reset(); self.sync_score = 50
                elif key == ord('c'): self.show_camera = not self.show_camera
                elif key == ord('a'): self.inhale_ratio = max(0.2, self.inhale_ratio - 0.05)
                elif key == ord('d'): self.inhale_ratio = min(0.8, self.inhale_ratio + 0.05)
                elif key == ord('f'):
                    if self.resonance_mode or self.pending_resonance_action is not None:
                        self.reset_resonance_finder()
                    else:
                        self.start_resonance_finder()
                is_up = (key == ord('w')) or (key in [82, 2490368, 0x260000])
                is_down = (key == ord('s')) or (key in [84, 2621440, 0x280000])
                if not self.resonance_mode:
                    if is_up: self.set_target_bpm(min(12.0, self.target_bpm + 0.5))
                    elif is_down: self.set_target_bpm(max(3.0, self.target_bpm - 0.5))
        finally:
            print("Exiting...")
            self.hr_monitor.shutdown()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    app.run()
