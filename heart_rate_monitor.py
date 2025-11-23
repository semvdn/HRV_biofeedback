import cv2
from collections import deque
import threading
import math
import time

from bluetooth_hr_monitor import BleakHeartRateMonitor


class HeartRateMonitor:
    """Manages heart rate and HRV data from a BLE device."""

    def __init__(self, buffer_size=300):
        self.ble_tracker = BleakHeartRateMonitor()
        self.ble_thread = threading.Thread(
            target=self.ble_tracker.run, daemon=True
        )
        self.ble_thread.start()
        self.bpm_history = deque(maxlen=buffer_size)
        self.hrv_history = deque(maxlen=buffer_size)

    def update(self):
        """Updates the BPM and HRV history from the BLE device."""
        current_bpm = self.get_bpm()
        if current_bpm > 0:
            self.bpm_history.append(current_bpm)
        elif len(self.bpm_history) > 0:
            # Hold last known BPM if we momentarily lose signal.
            self.bpm_history.append(self.bpm_history[-1])
        else:
            # Default initial value.
            self.bpm_history.append(60.0)
        
        hrv_rmssd_ms = self.get_hrv_rmssd_ms(window_sec=60.0)
        if hrv_rmssd_ms > 0:
            self.hrv_history.append(hrv_rmssd_ms)
        elif len(self.hrv_history) > 0:
            self.hrv_history.append(self.hrv_history[-1])
        else:
            self.hrv_history.append(0)


    def get_bpm(self):
        """Returns the current BPM from the Bluetooth device.

        Returns:
            int: The current BPM.
        """
        return self.ble_tracker.bpm

    # -------- IBI accessors for HRV from Bluetooth -------- #

    def get_ibi_ms(self):
        """Returns the most recent batch of IBIs from the Bluetooth device.

        Returns:
            list: A list of IBI values in milliseconds.
        """
        if self.ble_tracker.is_active:
            return self.ble_tracker.last_ibi_ms
        return []

    @property
    def ibi_history(self):
        """Returns the rolling history of IBIs from the Bluetooth device.

        Returns:
            list: A list of IBI values in milliseconds.
        """
        return self.ble_tracker.ibi_history if self.ble_tracker.is_active else []

    @property
    def ibi_status_history(self):
        """Returns the rolling history of IBI status codes.

        Returns:
            list: A list of IBI status codes.
        """
        return (
            self.ble_tracker.ibi_status_history
            if self.ble_tracker.is_active
            else []
        )

    def get_hrv_rmssd_ms(
        self,
        window_sec: float = 60.0,
        max_gap_sec: float = 3.0,
        use_status_filter: bool = True,
    ) -> float:
        """Computes the RMSSD of HRV from Bluetooth IBIs.

        Args:
            window_sec (float): The time window in seconds to consider for the calculation.
            max_gap_sec (float): The maximum allowed gap in seconds between IBIs.
            use_status_filter (bool): Whether to use IBI status codes for filtering.

        Returns:
            float: The RMSSD value in milliseconds.
        """
        # Need an active BLE stream with IBI history
        if not self.ble_tracker.is_active:
            return 0.0

        ibis = getattr(self.ble_tracker, "ibi_history", [])
        times = getattr(self.ble_tracker, "ibi_time_history", [])
        statuses = getattr(self.ble_tracker, "ibi_status_history", [])

        if not ibis or not times or len(ibis) != len(times):
            return 0.0

        now = time.time()

        # ------------------------------------------------------------------
        # Step 1: basic filtering by time window, amplitude, and status code.
        # ------------------------------------------------------------------
        candidates = []
        for idx, (t, v) in enumerate(zip(times, ibis)):
            # Time window
            if (now - t) > window_sec:
                continue

            # Physiologically plausible range (very conservative)
            if not (300.0 <= v <= 2000.0):
                continue

            # Get status if available, otherwise treat as 0
            st = 0
            if statuses and idx < len(statuses):
                try:
                    st = int(statuses[idx])
                except Exception:
                    st = 0

            # If we honour status flags, drop anything not 0 or -1
            if use_status_filter and st not in (0, -1):
                continue

            candidates.append((t, v, st))

        if len(candidates) < 3:
            return 0.0

        # Sort chronologically
        candidates.sort(key=lambda x: x[0])

        # ------------------------------------------------------------------
        # Step 2: segment by gaps and reject outliers within each segment.
        # ------------------------------------------------------------------
        diffs_sq = []

        # Start with the first sample as the reference
        prev_t, prev_v, prev_st = candidates[0]

        for t, v, st in candidates[1:]:
            # Break segments on large time gaps
            if (t - prev_t) > max_gap_sec:
                prev_t, prev_v, prev_st = t, v, st
                continue

            # Beat-to-beat change
            dv = v - prev_v
            ratio = v / prev_v if prev_v > 0 else 1.0

            if not use_status_filter:
                # Ignore status codes, just generic outlier check
                ok = (0.3 <= ratio <= 3.0) and (abs(dv) <= 500.0)
            else:
                if st == 0:
                    # "Good" beats: allow broader changes
                    ok = (0.3 <= ratio <= 3.0) and (abs(dv) <= 500.0)
                elif st == -1:
                    # "Uncertain" beats: only accept if very close to previous
                    # (conservative use of -1)
                    ok = (0.8 <= ratio <= 1.25) and (abs(dv) <= 150.0)
                else:
                    ok = False

            if ok:
                # Accept this beat-to-beat interval for RMSSD
                diffs_sq.append(dv * dv)
                prev_t, prev_v, prev_st = t, v, st
            # If not ok: skip this sample and keep the previous accepted one
            # as the reference for the next candidate.

        if len(diffs_sq) < 2:
            return 0.0

        rmssd = math.sqrt(sum(diffs_sq) / len(diffs_sq))
        return float(rmssd)

    # ------------------------------------------------------ #

    def get_active_source(self):
        """Returns the active source of HR data.

        Returns:
            str: The active source, which is always "Bluetooth".
        """
        return "Bluetooth"

    def draw_debug(self, frame):
        """Draws debugging information on the frame.

        Args:
            frame (np.ndarray): The frame to draw on.

        Returns:
            np.ndarray: The frame with debugging information.
        """
        status_text = f"BT: {self.ble_tracker.status}"
        color = (0, 255, 0) if self.ble_tracker.is_active else (255, 100, 100)
        cv2.putText(
            frame,
            status_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            status_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )
        return frame

    def shutdown(self):
        """Shuts down the Bluetooth monitor."""
        print("Shutting down Bluetooth monitor...")
        self.ble_tracker.stop()
