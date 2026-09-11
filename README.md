# Autonomous RC Car with PilotNet CNN

An end-to-end autonomous RC car built using a **Raspberry Pi 4B**, **Pi Camera Module Rev 1.3**, and a **PilotNet CNN**. The car learns steering behavior from manually collected driving data and performs real-time autonomous steering on a lane-marked track.

## Project Overview

The system follows an end-to-end autonomous driving pipeline:

**Camera → Image Preprocessing → PilotNet CNN → Steering Angle → Servo Motor**

A Raspberry Pi 4B captures images from the camera, preprocesses them, runs the trained neural network using **ONNX Runtime**, and sends the predicted steering angle to the steering servo through hardware PWM.

The Raspberry Pi also hosts a **Flask-based web interface** over its own WiFi hotspot for manual driving, live camera streaming, data collection, autonomous mode, LED control, speed control, and dataset download.

## Features

- Manual RC car control through a mobile-friendly web interface
- Live MJPEG camera streaming
- Training-data collection with steering-angle labels
- PilotNet CNN for end-to-end steering prediction
- Real-time autonomous steering at approximately 20 Hz
- ONNX model deployment using ONNX Runtime
- Hardware-timed servo PWM using pigpio
- Self-hosted WiFi hotspot — no external router required
- Automatic startup using a systemd service
- Motor speed control and direction control
- Front LED control
- Dataset download as a ZIP archive

## Hardware

| Component | Description |
|---|---|
| Raspberry Pi 4B | Main processing unit |
| Raspberry Pi Camera Module Rev 1.3 | Visual perception |
| MG90S Micro Servo | Steering control |
| L298N Motor Driver | DC motor control |
| 7.4V LiPo Battery Pack | Motor and servo power |
| 5V/2A Power Bank | Raspberry Pi power |

The Raspberry Pi and motor/servo use separate power supplies, while their grounds are connected through the L298N ground terminal.

## GPIO Pin Mapping

| Function | GPIO (BCM) | Physical Pin |
|---|---:|---:|
| Servo Signal | GPIO 12 | Pin 32 |
| Motor IN1 | GPIO 13 | Pin 33 |
| Motor IN2 | GPIO 14 | Pin 8 |
| Motor ENA (PWM) | GPIO 19 | Pin 35 |
| Front LED | GPIO 24 | Pin 18 |
| 5V Power | — | Pin 2 or 4 |
| Common GND | — | Pin 6 |

## Software Stack

- Raspberry Pi OS Bookworm 64-bit
- Python 3.13
- Flask
- Picamera2
- OpenCV
- NumPy
- pigpio
- ONNX Runtime
- TensorFlow/Keras
- tf2onnx
- Google Colab

## Data Collection

Training data was collected by manually driving the RC car around a track made using blue masking tape on a white tiled floor.

### Image Processing

- Original camera capture: **1280 × 720**
- CNN input: **200 × 66 grayscale**
- CLAHE contrast enhancement
- Capture rate: **10 FPS**
- JPEG image storage
- Steering angle stored in the filename

Example:

```text
frame_00001_angle_90.jpg
frame_00145_angle_135.jpg
frame_00312_angle_45.jpg
```

Steering labels:

- **90°** — Straight
- **135°** — Left
- **45°** — Right

### Dataset

- 7 driving sessions
- 16,223 total frames
- Straight frames downsampled to reduce class imbalance
- 80% training / 20% validation split

## PilotNet CNN

The project uses the **PilotNet** architecture for end-to-end steering prediction.

The network takes a normalized grayscale image of shape:

```text
(66, 200, 1)
```

and predicts a continuous steering angle between:

```text
45° and 135°
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss | Mean Squared Error (MSE) |
| Metric | Mean Absolute Error (MAE) |
| Batch Size | 32 |
| Maximum Epochs | 100 |
| Early Stopping Patience | 10 |
| Platform | Google Colab GPU T4 |

Training completed after approximately **49 epochs** due to early stopping.

The validation loss decreased from approximately **3,695 to 431**, an improvement of about **88%**.

## Model Conversion and Deployment

The trained Keras model was converted to **ONNX** using `tf2onnx`.

This was done because the full TensorFlow environment required significantly more storage than was practical for the Raspberry Pi's 8GB SD card.

The resulting ONNX model is approximately **3–7 MB**, while ONNX Runtime requires approximately **15 MB**.

The deployed model uses the input signature:

```text
(None, 66, 200, 1)
```

## Autonomous Inference Pipeline

During autonomous operation, the Raspberry Pi performs the following steps:

1. Capture a frame using Picamera2.
2. Resize the image to 200 × 66.
3. Convert the image to grayscale.
4. Apply CLAHE contrast enhancement.
5. Normalize pixel values to 0–1.
6. Convert the image into a float32 input tensor.
7. Run inference using ONNX Runtime.
8. Clamp the predicted angle to 45°–135°.
9. Convert the angle into a servo pulse width.
10. Send the PWM signal to the steering servo through GPIO 12.

The autonomous control loop operates at approximately **20 Hz**.

### Steering Mapping

The predicted steering angle is converted to a servo pulse width using:

```text
pulse = 500 + ((angle - 45) / 90) × 2000 μs
```

## Web Interface

The Flask web application provides:

- Forward/reverse/left/right controls
- Hold-to-drive control
- Live camera stream
- Data collection mode
- Frame counter
- Autonomous mode toggle
- LED control
- Motor speed slider
- Dataset download

The Raspberry Pi creates its own WiFi hotspot:

```text
SSID: RC-CAR-Pi
IP:   10.42.0.1
Port: 5000
```

No external router is required.

## Automatic Startup

A systemd service named:

```text
rccar.service
```

starts the complete system automatically when the Raspberry Pi boots.

The startup script:

1. Starts the pigpio daemon.
2. Disconnects from external WiFi networks.
3. Activates the RC-CAR-Pi hotspot.
4. Starts the Flask application.

The car can therefore become operational without a monitor, keyboard, or external router.

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Servo jitter | Used pigpio hardware-timed PWM |
| WiFi hotspot instability | Configured NetworkManager connection priority |
| TensorFlow too large for SD card | Converted model to ONNX |
| Python 3.13 compatibility | Used ONNX Runtime |
| Camera overexposure | Applied CLAHE and exposure controls |
| GPIO conflicts during boot | Added GPIO cleanup and systemd ordering |
| Duplicate hotspot on boot | Separated hotspot creation and activation |

## Results

The completed system successfully demonstrated:

- Functional manual control through the web interface
- Real-time camera streaming
- Successful collection of **16,223 labeled frames**
- Successful PilotNet training
- **88% reduction in validation loss**
- Successful Keras-to-ONNX model conversion
- Successful ONNX Runtime inference on Raspberry Pi 4B
- Real-time autonomous steering through the servo
- Automatic startup and self-hosted WiFi operation

## Future Improvements

- Add data augmentation for better generalization
- Add ultrasonic sensing for obstacle detection and emergency stopping
- Add a PID controller for smoother steering
- Upgrade to Raspberry Pi Camera Module 3
- Test more complex tracks with intersections and tighter curves
- Explore quantization-aware training to reduce inference latency

## Project Team

- Azeem Ashraf
- Muhammad Umer Shahzad
- Syed Aneeq Ahmad
- Hafiz Muzammil Hussain

## Course

**EC-310 Microprocessor and Microcontrollers**

**B.E. Computer Engineering**

## License

This project is intended for educational and research purposes.
