# HRV Biofeedback Trainer

This application is a real-time Heart Rate Variability (HRV) biofeedback trainer. It provides a visual pacer to guide your breathing and uses data from a Bluetooth heart rate monitor to track your physiological response. The goal is to help you find your resonant breathing frequency, which can maximize HRV and promote a state of calm and focus.



https://github.com/user-attachments/assets/99858e3f-5e28-4aee-be87-f299e9247d87



## Features

- **Bluetooth Heart Rate Integration**: Uses a dedicated Bluetooth Low Energy (BLE) heart rate monitor (like the companion watch app) for accurate Inter-Beat Interval (IBI) data.
- **Breathing Pacer**: A fluid, visually engaging pacer guides you to breathe at a target rate.
- **Real-time Biofeedback**: Tracks your breathing motion via the webcam and compares it to the pacer, providing a "Rhythm Match" score.
- **HRV Analysis**: Calculates RMSSD (Root Mean Square of Successive Differences), a standard time-domain measure of HRV, from the BLE device's IBI data.
- **Resonance Finder**: An automated mode that sweeps through different breathing rates (e.g., 6.5, 6.0, 5.5, 5.0, 4.5 BPM) to find the frequency that produces the highest HRV amplitude for you.
- **Comprehensive Dashboard**: Displays real-time graphs of your breathing and heart rate, along with key metrics like pacer rate, breathing rate, heart rate, and HRV.
- **Companion Wear OS App**: Works with the provided `HRVB_Companion_app` to stream heart rate and IBI data directly from a Wear OS watch.

## How It Works

The application is built around three main components:

1.  **`breathing_monitor.py`**: This is the main application file. It uses OpenCV to capture video from the webcam to track the subtle movements of your chest and shoulders to estimate your breathing rate and phase (inhale/exhale). It also runs the main UI and biofeedback dashboard.
2.  **`heart_rate_monitor.py`**: This module manages the heart rate data, exclusively using data from a connected BLE device.
3.  **`bluetooth_hr_monitor.py`**: This script handles the Bluetooth connection in a separate thread. It scans for and connects to devices advertising the custom Heart Rate service provided by the companion app.

## Companion Watch App

A companion Wear OS application (`HRVB_Companion_app/app-release.apk`) is included. This app turns your watch into a dedicated HRV sensor, broadcasting detailed heart rate and IBI data over a custom BLE service. This is the required method for accurate HRV readings.

For installation instructions, please see the [HRVB_Companion_app/README.md](HRVB_Companion_app/README.md).

## Setup and Installation

### Prerequisites

- Python 3.8+
- A webcam
- (Recommended) A Wear OS watch with the companion app installed

### Installation

1.  **Clone the repository or download the files.**

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Download the Haar cascade:**
    ```bash
    curl -L -o haarcascade_frontalface_default.xml https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
    ```
    On PowerShell:
    ```powershell
    Invoke-WebRequest -Uri "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml" -OutFile "haarcascade_frontalface_default.xml"
    ```

4.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Make sure your webcam is connected.
2.  If using a BLE device, ensure it is turned on and ready to connect. If using the watch app, open it on your watch.
3.  Run the main application from your terminal:
    ```bash
    python breathing_monitor.py
    ```

A window will appear displaying the biofeedback dashboard.

## Controls

- **[UP/W]**: Increase pacer breathing rate.
- **[DOWN/S]**: Decrease pacer breathing rate.
- **[D]**: Increase the inhale duration ratio.
- **[A]**: Decrease the inhale duration ratio.
- **[F]**: Start/Stop the automatic Resonance Finder sweep.
- **[R]**: Reset the breathing tracker.
- **[C]**: Toggle the camera view on/off.
- **[Q] or [ESC]**: Quit the application.

## AI Assistance

Some scripting and refactors were produced with help from Google's Gemini 3.0 Pro and Openai's GPT 5.1-codex-high.
