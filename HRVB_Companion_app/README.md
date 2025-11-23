# HRVB Companion Wear OS App

## Description

This Wear OS application acts as a bridge to broadcast heart rate and heart rate variability (HRV) data over Bluetooth Low Energy (BLE). It reads sensor data from the watch and exposes it through a custom BLE service, allowing other devices to connect and receive real-time cardiac information.

The app provides a custom BLE characteristic for detailed HRV data, including inter-beat interval (IBI) measurements.

## How it Works

The application utilizes the watch's built-in heart rate sensor. It sets up a BLE GATT server that advertises a custom HRV Service (UUID `0000ffb0-0000-1000-8000-00805f9b34fb`).

Once a client device connects and subscribes to notifications, the app sends data through a custom characteristic that provides a detailed payload with HR, HR status, and a list of recent IBI values in milliseconds.

## Data Sent

The application sends data over a custom BLE characteristic:

*   **Custom HRV Service (0000ffb0-0000-1000-8000-00805f9b34fb)**
    *   **HRV Data (0000ffb2-0000-1000-8000-00805f9b34fb)**:
        *   Heart Rate (uint8, BPM)
        *   HR Status (int8)
        *   IBI Count (uint8)
        *   IBI Data (array of {IBI (uint16, ms), Status (int8)})

## Installation

To install the application on your Wear OS device, you can use the Android Debug Bridge (ADB).

1.  **Enable ADB Debugging** on your watch from the Developer Options in the settings.
2.  **Connect your watch** to your computer (either via USB or Wi-Fi).
3.  **Install the APK** using the following ADB command:

    ```bash
    adb install path/to/your/app-release.apk
    ```

