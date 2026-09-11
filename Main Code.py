# ============================================================
# FILE: app.py
# PROJECT: Autonomous RC Car — Raspberry Pi 4B
# SERVO: pigpio hardware PWM — zero jitter guaranteed
# MODEL: ONNX Runtime — no TensorFlow needed
# ============================================================

from flask import Flask, Response, jsonify, render_template, request, send_file
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, H264Encoder
from picamera2.outputs import FileOutput
import RPi.GPIO as GPIO
import pigpio
import threading
import time
import io
import os
import cv2
import numpy as np
import zipfile
from datetime import datetime

app = Flask(__name__)

# ============================================================
# PIGPIO SETUP — hardware PWM for servo, zero jitter
# ============================================================
pi = pigpio.pi()
if not pi.connected:
    print("ERROR: pigpiod not running — run: sudo pigpiod")
    exit()

SERVO_PIN = 12

# ============================================================
# SERVO PULSE WIDTHS in microseconds
# MG90S: 500-2500us range
# Adjust LEFT/RIGHT for sharper turns
# ============================================================
SERVO_CENTRE_US = 1500
SERVO_LEFT_US   = 2500
SERVO_RIGHT_US  =  500

def servo_set(pulse_us):
    pulse_us = max(500, min(2500, pulse_us))
    pi.set_servo_pulsewidth(SERVO_PIN, pulse_us)

def servo_stop():
    pi.set_servo_pulsewidth(SERVO_PIN, 0)

# ============================================================
# GPIO PIN DEFINITIONS — motor and LED
# ============================================================
MOTOR_IN1 = 13
MOTOR_IN2 = 14
MOTOR_ENA = 19
LED_PIN   = 24

# ============================================================
# GPIO SETUP
# ============================================================
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(LED_PIN,   GPIO.OUT)

motor_pwm = GPIO.PWM(MOTOR_ENA, 800)
motor_pwm.start(0)

GPIO.output(MOTOR_IN1, GPIO.LOW)
GPIO.output(MOTOR_IN2, GPIO.LOW)
GPIO.output(LED_PIN,   GPIO.LOW)

# ============================================================
# STATE
# ============================================================
motor_active  = False
steer_active  = False
current_steer = 'centre'
led_on        = False
motor_speed   = 70
last_motor    = 0.0
last_steer    = 0.0
TIMEOUT       = 0.25

# ============================================================
# DATA COLLECTION STATE
# ============================================================
collecting       = False
collect_dir      = ""
frame_count      = 0
collect_lock     = threading.Lock()

# ============================================================
# AUTONOMOUS STATE
# ============================================================
autonomous  = False
model       = None
input_name  = None
model_path  = os.path.expanduser("~/RC_Car/model/pilotnet.onnx")

# ============================================================
# RECORDING STATE
# ============================================================
recording        = False
record_encoder   = None
recordings_dir   = os.path.expanduser("~/RC_Car/recordings")
dataset_base_dir = os.path.expanduser("~/RC_Car/dataset")
os.makedirs(recordings_dir,   exist_ok=True)
os.makedirs(dataset_base_dir, exist_ok=True)

# ============================================================
# MOTOR FUNCTIONS
# ============================================================
def motor_stop():
    global motor_active
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(0)
    motor_active = False

def motor_forward():
    global motor_active, last_motor
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(motor_speed)
    motor_active = True
    last_motor   = time.time()

def motor_backward():
    global motor_active, last_motor
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(motor_speed)
    motor_active = True
    last_motor   = time.time()

# ============================================================
# STEERING — pigpio hardware PWM, instant and jitter-free
# ============================================================
def steer_left():
    global steer_active, last_steer, current_steer
    servo_set(SERVO_LEFT_US)
    steer_active  = True
    current_steer = 'left'
    last_steer    = time.time()

def steer_right():
    global steer_active, last_steer, current_steer
    servo_set(SERVO_RIGHT_US)
    steer_active  = True
    current_steer = 'right'
    last_steer    = time.time()

def steer_centre():
    global steer_active, current_steer
    servo_set(SERVO_CENTRE_US)
    steer_active  = False
    current_steer = 'centre'

# ============================================================
# CAMERA SETUP
# ============================================================
picam  = Picamera2()
config = picam.create_video_configuration(
    main    = {"size": (1280, 720), "format": "BGR888"},
    lores   = {"size": (640,  480), "format": "YUV420"},
    display = "lores"
)
picam.configure(config)
picam.start()
time.sleep(1)

# Fix overexposure
picam.set_controls({
    "AeEnable":       True,
    "AwbEnable":      True,
    "Brightness":    -0.2,
    "Contrast":       1.5,
    "ExposureValue": -1.0,
})
time.sleep(0.5)
print("Camera started OK")

# ============================================================
# MJPEG STREAM
# ============================================================
class StreamOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame     = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

stream_output  = StreamOutput()
stream_encoder = MJPEGEncoder(bitrate=2000000)
picam.start_encoder(stream_encoder, FileOutput(stream_output), name="lores")

def generate_frames():
    while True:
        try:
            with stream_output.condition:
                stream_output.condition.wait(timeout=3.0)
                frame = stream_output.frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)

# ============================================================
# DATA COLLECTION
# Saves 200x66 grayscale frames with angle in filename
# ============================================================
def data_collection_loop():
    global collecting, frame_count
    while True:
        if collecting:
            try:
                frame   = picam.capture_array("main")
                resized = cv2.resize(frame, (200, 66))
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                # CLAHE contrast enhancement
                clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray    = clahe.apply(gray)

                if   current_steer == 'left':  angle = 135
                elif current_steer == 'right': angle = 45
                else:                          angle = 90

                fname = f"frame_{frame_count:05d}_angle_{angle}.jpg"
                fpath = os.path.join(collect_dir, fname)
                cv2.imwrite(fpath, gray)

                with collect_lock:
                    frame_count += 1

            except Exception as e:
                print(f"Collection error: {e}")
            time.sleep(0.1)
        else:
            time.sleep(0.05)

threading.Thread(target=data_collection_loop, daemon=True).start()

# ============================================================
# AUTONOMOUS MODE — ONNX Runtime inference
# ============================================================
def load_model():
    global model, input_name
    if not os.path.exists(model_path):
        print(f"No model at {model_path}")
        return False
    try:
        import onnxruntime as ort
        model      = ort.InferenceSession(model_path)
        input_name = model.get_inputs()[0].name
        print(f"ONNX model loaded OK — input: {input_name}")
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

def autonomous_loop():
    global autonomous
    while True:
        if autonomous and model is not None:
            try:
                # Capture frame
                frame   = picam.capture_array("main")
                resized = cv2.resize(frame, (200, 66))
                gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                # CLAHE — same preprocessing as training
                clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray    = clahe.apply(gray)

                # Prepare input — shape (1, 66, 200, 1) float32 normalized
                inp = gray.reshape(1, 66, 200, 1).astype(np.float32) / 255.0

                # Run ONNX inference
                predicted = float(model.run(None, {input_name: inp})[0][0][0])

                # Clamp predicted angle to valid range
                predicted = max(45.0, min(135.0, predicted))

                # Convert angle (45-135) to servo pulse (500-2500us)
                pulse = int(500 + ((predicted - 45) / 90.0) * 2000)
                pulse = max(500, min(2500, pulse))

                servo_set(pulse)
                motor_forward()

            except Exception as e:
                print(f"Autonomous error: {e}")
                motor_stop()
                steer_centre()
            time.sleep(0.05)
        else:
            time.sleep(0.05)

threading.Thread(target=autonomous_loop, daemon=True).start()

# ============================================================
# RECORDING
# ============================================================
def start_recording():
    global recording, record_encoder
    if recording:
        return "already_recording"
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath       = os.path.join(recordings_dir, f"drive_{timestamp}.h264")
    record_encoder = H264Encoder(bitrate=4000000)
    picam.start_encoder(record_encoder, FileOutput(filepath), name="main")
    recording = True
    return filepath

def stop_recording():
    global recording, record_encoder
    if not recording:
        return "not_recording"
    picam.stop_encoder(record_encoder)
    recording = False
    return "stopped"

# ============================================================
# SAFETY WATCHDOG
# ============================================================
def watchdog():
    while True:
        if not autonomous:
            now = time.time()
            if motor_active and (now - last_motor > TIMEOUT):
                motor_stop()
            if steer_active and (now - last_steer > TIMEOUT):
                steer_centre()
        time.sleep(0.05)

threading.Thread(target=watchdog, daemon=True).start()

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/cmd')
def command():
    global motor_speed, led_on, collecting
    global collect_dir, frame_count, autonomous
    cmd = request.args.get('action', '')

    if autonomous and cmd in ['forward', 'backward', 'left',
                              'right', 'centre', 'stopmotor']:
        return jsonify({'status': 'autonomous_mode'})

    if   cmd == 'forward':   motor_forward()
    elif cmd == 'backward':  motor_backward()
    elif cmd == 'stopmotor': motor_stop()
    elif cmd == 'left':      steer_left()
    elif cmd == 'right':     steer_right()
    elif cmd == 'centre':    steer_centre()
    elif cmd == 'stop':      motor_stop(); steer_centre()

    elif cmd == 'led':
        led_on = not led_on
        GPIO.output(LED_PIN, GPIO.HIGH if led_on else GPIO.LOW)

    elif cmd == 'record_start':
        path = start_recording()
        return jsonify({'status': 'recording', 'file': path})

    elif cmd == 'record_stop':
        stop_recording()
        return jsonify({'status': 'stopped'})

    elif cmd == 'collect_start':
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        collect_dir = os.path.join(dataset_base_dir, f"session_{timestamp}")
        os.makedirs(collect_dir, exist_ok=True)
        frame_count = 0
        collecting  = True
        return jsonify({'status': 'collecting', 'dir': collect_dir})

    elif cmd == 'collect_stop':
        collecting = False
        return jsonify({'status': 'stopped', 'frames': frame_count})

    elif cmd == 'auto_start':
        if model is None:
            ok = load_model()
            if not ok:
                return jsonify({'status': 'error',
                                'msg': 'No model at ~/RC_Car/model/pilotnet.onnx'})
        autonomous = True
        return jsonify({'status': 'autonomous'})

    elif cmd == 'auto_stop':
        autonomous = False
        motor_stop()
        steer_centre()
        return jsonify({'status': 'manual'})

    elif cmd.startswith('speed_'):
        pct = int(cmd.split('_')[1])
        motor_speed = 0 if pct < 30 else int(30 + (pct - 30))

    return jsonify({'status': 'ok', 'cmd': cmd})

@app.route('/download_dataset')
def download_dataset():
    zip_path = os.path.expanduser("~/RC_Car/dataset.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(dataset_base_dir):
            for file in files:
                filepath = os.path.join(root, file)
                zipf.write(filepath,
                           os.path.relpath(filepath, dataset_base_dir))
    return send_file(zip_path, as_attachment=True,
                     download_name='dataset.zip')

@app.route('/status')
def status():
    return jsonify({
        'recording':  recording,
        'collecting': collecting,
        'autonomous': autonomous,
        'frames':     frame_count,
        'led':        led_on,
        'speed':      motor_speed
    })

# ============================================================
# CLEANUP
# ============================================================
import atexit

def cleanup():
    motor_stop()
    steer_centre()
    time.sleep(0.2)
    servo_stop()
    pi.stop()
    motor_pwm.stop()
    GPIO.cleanup()

atexit.register(cleanup)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    steer_centre()
    print("=" * 40)
    print("RC Car server starting...")
    print("Open: http://10.42.0.1:5000")
    print("=" * 40)
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        cleanup()
