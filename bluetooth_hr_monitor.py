import asyncio
import struct
import time
import threading
from collections import deque
from typing import List, Optional

from bleak import BleakScanner, BleakClient

# Custom HRV service/characteristic UUIDs (must match the watch app)
HRV_SERVICE_UUID = "0000ffb0-0000-1000-8000-00805f9b34fb"
HRV_DATA_CHAR_UUID = "0000ffb2-0000-1000-8000-00805f9b34fb"


class BleakHeartRateMonitor:
    """Manages BLE connection and data retrieval for HR and IBI."""

    def __init__(self):
        self.lock = threading.Lock()

        self._bpm: int = 0
        self._hr_status: int = 0
        self._status: str = "Initializing..."
        self._is_connected: bool = False
        self._last_update_time: float = 0.0

        # Most recent batch of IBIs (ms) and their status codes
        self._last_ibi_ms: List[float] = []
        self._last_ibi_status: List[int] = []

        # IBI value / status history, with timestamps (seconds since epoch)
        self._ibi_history: deque[float] = deque(maxlen=2048)
        self._ibi_time_history: deque[float] = deque(maxlen=2048)
        self._ibi_status_history: deque[int] = deque(maxlen=2048)

        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bpm(self) -> int:
        """Returns the latest BPM value."""
        with self.lock:
            return self._bpm

    @property
    def hr_status(self) -> int:
        """Returns the latest heart rate status from the watch."""
        with self.lock:
            return self._hr_status

    @property
    def status(self) -> str:
        """Returns the current status of the BLE connection."""
        with self.lock:
            return self._status

    @property
    def is_active(self) -> bool:
        """Returns True if the BLE device is connected and sending data."""
        with self.lock:
            return self._is_connected and (time.time() - self._last_update_time) < 5.0

    @property
    def last_ibi_ms(self) -> List[float]:
        """Returns the last batch of IBI values in milliseconds."""
        with self.lock:
            return list(self._last_ibi_ms)

    @property
    def last_ibi_status(self) -> List[int]:
        """Returns the last batch of IBI status codes."""
        with self.lock:
            return list(self._last_ibi_status)

    @property
    def ibi_history(self) -> List[float]:
        """Returns the rolling history of all IBI values."""
        with self.lock:
            return list(self._ibi_history)

    @property
    def ibi_time_history(self) -> List[float]:
        """Returns the rolling history of IBI timestamps."""
        with self.lock:
            return list(self._ibi_time_history)

    @property
    def ibi_status_history(self) -> List[int]:
        """Returns the rolling history of IBI status codes."""
        with self.lock:
            return list(self._ibi_status_history)

    # ------------------------------------------------------------------
    # Internal BLE logic
    # ------------------------------------------------------------------

    async def _find_device(self):
        """Scans for and returns a BLE device that advertises the custom HRV service."""
        with self.lock:
            self._status = "Scanning for HRV service..."
        print("BT Status: Scanning for HRV service...")

        def _filter_hrv_service(device, adv_data):
            uuids = adv_data.service_uuids or []
            for u in uuids:
                if u.lower() == HRV_SERVICE_UUID:
                    return True
            return False

        device = await BleakScanner.find_device_by_filter(
            _filter_hrv_service, timeout=10.0
        )
        if device is not None:
            with self.lock:
                self._status = f"Found device: {device.name or device.address}"
            print(f"Found HRV device: {device.name or device.address}")
            return device

        with self.lock:
            self._status = "No BLE device with HRV service found"
        print("No BLE device with HRV service found")
        return None

    def _handle_hrv_data(self, handle: int, data: bytearray):
        """Handles notifications from the custom HRV characteristic."""
        if not data or len(data) < 3:
            return

        heart_rate = data[0]
        hr_status = struct.unpack_from("<b", data, 1)[0]
        count = data[2]

        offset = 3
        ibi_ms_list: List[float] = []
        ibi_status_list: List[int] = []

        for _ in range(count):
            if len(data) < offset + 3:
                break
            ibi_ms = struct.unpack_from("<H", data, offset)[0]
            status = struct.unpack_from("<b", data, offset + 2)[0]
            offset += 3

            if ibi_ms > 0:
                ibi_ms_list.append(float(ibi_ms))
                ibi_status_list.append(int(status))

        now_ts = time.time()

        with self.lock:
            self._bpm = int(heart_rate)
            self._hr_status = int(hr_status)
            self._last_update_time = now_ts
            self._is_connected = True
            self._status = f"Connected ({self._bpm} BPM, hrStatus={self._hr_status})"

            self._last_ibi_ms = ibi_ms_list
            self._last_ibi_status = ibi_status_list

            for v, st in zip(ibi_ms_list, ibi_status_list):
                self._ibi_history.append(v)
                self._ibi_time_history.append(now_ts)
                self._ibi_status_history.append(st)

    async def _get_hrv_char_handle(self, client: BleakClient) -> Optional[int]:
        """Finds the handle for the custom HRV characteristic."""
        services = getattr(client, "services", None)

        # For newer Bleak versions, services may need to be explicitly discovered
        if services is None:
            get_services = getattr(client, "get_services", None)
            if callable(get_services):
                services = await get_services()

        if services is None:
            return None

        for service in services:
            for ch in service.characteristics:
                if not ch.uuid:
                    continue
                if ch.uuid.lower() != HRV_DATA_CHAR_UUID:
                    continue
                props = getattr(ch, "properties", []) or []
                if "notify" in props or "indicate" in props:
                    return ch.handle

        return None

    async def _run_client(self, device):
        """Connects to a BLE device and listens for HRV notifications."""
        with self.lock:
            self._status = "Connecting..."
        print(f"Connecting to {device.address}...")
        async with BleakClient(device, timeout=20.0) as client:
            # Handle both "property" and "callable" variants of is_connected
            is_conn_attr = client.is_connected
            if callable(is_conn_attr):
                maybe = is_conn_attr()
                if asyncio.iscoroutine(maybe):
                    is_connected = await maybe
                else:
                    is_connected = bool(maybe)
            else:
                is_connected = bool(is_conn_attr)

            if not is_connected:
                with self.lock:
                    self._status = "Connection failed"
                print(f"Connection to {device.address} failed")
                return

            with self.lock:
                self._is_connected = True
                self._status = f"Connected to {device.name or device.address}"
            print(f"Connected to {device.name or device.address}")

            # Find the exact handle for the custom HRV characteristic
            hrv_handle = await self._get_hrv_char_handle(client)
            if hrv_handle is None:
                with self.lock:
                    self._status = "No HRV characteristic found"
                print("ERROR: No HRV characteristic found on device.")
                return

            # Subscribe using the HANDLE
            await client.start_notify(hrv_handle, self._handle_hrv_data)

            try:
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.1)
            finally:
                # Stop notifications safely
                try:
                    await client.stop_notify(hrv_handle)
                except Exception:
                    pass
                with self.lock:
                    self._is_connected = False
                    self._bpm = 0
                    self._hr_status = 0
                    self._last_ibi_ms = []
                    self._last_ibi_status = []
                    self._status = "Disconnected"
                print("BT Status: Disconnected")

    async def _main_loop(self):
        """The main async loop for the BLE client."""
        while not self._stop_event.is_set():
            device = await self._find_device()
            if device is None:
                # Wait before rescanning
                await asyncio.sleep(5.0)
                continue

            try:
                await self._run_client(device)
            except Exception as e:
                with self.lock:
                    self._status = f"Error: {e!r}"
                print(f"BT Error: {e!r}")
            finally:
                if not self._stop_event.is_set():
                    await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def run(self):
        """Runs the main async loop in a background thread."""
        asyncio.run(self._main_loop())

    def stop(self):
        """Stops the background thread."""
        self._stop_event.set()
