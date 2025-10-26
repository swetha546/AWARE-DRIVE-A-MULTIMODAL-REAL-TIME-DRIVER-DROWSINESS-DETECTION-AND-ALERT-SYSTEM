
import cv2
import dlib
import numpy as np
import time
import threading
from scipy.spatial import distance as dist
import math
from collections import deque
import csv
import os
import sys

# ---------------- CONFIG ----------------
ROLLING_WINDOW = 60            # seconds
ALERT_COOLDOWN = 8             # seconds (normal cooldown)
ESCALATION_COOLDOWN = 4        # seconds between escalation steps
YAWN_MAR_THRESHOLD = 0.70
DEFAULT_EAR_THRESHOLD = 0.20
NOD_DIFF_THRESHOLD_DEG = 7
NOD_COOLDOWN = 1.5
CALIBRATE_SECONDS = 4
LOG_CSV = "aware_drive_log.csv"
ALARM_PATH = "alarm.wav"           # gentle alarm
LOUD_ALARM_PATH = "loud_alarm.wav" # loud alarm for escalation
USE_DEEPFACE = True
PRETEND_SLEEP_SECONDS = 2.0        # eyes closed duration to count as sleeping/pretend
# ----------------------------------------

# Try optional DeepFace
try:
    if USE_DEEPFACE:
        from deepface import DeepFace
    else:
        DeepFace = None
except Exception:
    DeepFace = None

# ---------------- Utilities ----------------
def async_speak(message):
    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()

def async_play_sound(path):
    def run():
        try:
            from playsound import playsound
            playsound(path)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return 0.0 if C == 0 else (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return 0.0 if C == 0 else (A + B) / (2.0 * C)

def head_angle(left, right):
    dy = right[1] - left[1]
    dx = right[0] - left[0]
    return math.degrees(math.atan2(dy, dx))

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255.0

def get_heart_rate():
    """
    Hook for real HR sensor:
    Replace this function to read PPG/ECG via serial/Bluetooth/USB and return bpm integer.
    """
    return int(np.random.randint(58, 85))

def detect_emotion(frame, face_coords):
    if DeepFace is None:
        return 0.0
    try:
        x, y, w, h = face_coords
        face_img = frame[max(0,y):y+h, max(0,x):x+w]
        if face_img.size == 0:
            return 0.0
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotions = analysis[0]["emotion"] if isinstance(analysis, list) else analysis.get("emotion", {})
        score = emotions.get("sad", 0) + 0.5 * emotions.get("neutral", 0)
        return float(score) / 100.0
    except Exception:
        return 0.0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def scaled_score(value, baseline, scale=20.0):
    # For EAR: baseline - value ; higher -> more fatigue
    diff = baseline - value
    return float(sigmoid(scale * diff))

# --------------- Behavior Tracker ----------------
class BehaviorTracker:
    def __init__(self, window_secs=ROLLING_WINDOW):
        self.timestamps = deque()
        self.ear_vals = deque()
        self.mar_vals = deque()
        self.angle_vals = deque()
        self.blinks = deque()
        self.yawns = deque()
        self.emotions = deque()
        self.hrs = deque()
        self.angle_history = deque(maxlen=20)
        self.nod_count = 0
        self.nod_last_time = 0.0
        self.log = []
        self.window_secs = window_secs
        self.eye_closed_since = None  # to detect prolonged eye closure
        self.last_movement_time = time.time()

    def add(self, timestamp, ear, mar, angle, blink, yawn, emotion, hr):
        self.timestamps.append(timestamp)
        self.ear_vals.append(ear)
        self.mar_vals.append(mar)
        self.angle_vals.append(angle)
        self.blinks.append(blink)
        self.yawns.append(yawn)
        self.emotions.append(emotion)
        self.hrs.append(hr)
        self.angle_history.append(angle)
        self.detect_nod(timestamp)
        self._prune_old(timestamp)
        # update movement time (if ear or angle changed)
        if len(self.ear_vals) >= 2:
            if abs(self.ear_vals[-1] - self.ear_vals[-2]) > 0.01 or abs(self.angle_history[-1] - self.angle_history[-2]) > 0.5:
                self.last_movement_time = timestamp
        self.log.append({
            "timestamp": timestamp,
            "ear": ear, "mar": mar, "angle": angle,
            "blink": blink, "yawn": yawn,
            "emotion": emotion, "hr": hr,
            "nod_count": self.nod_count
        })

    def _prune_old(self, now):
        while self.timestamps and now - self.timestamps[0] > self.window_secs:
            self.timestamps.popleft()
            self.ear_vals.popleft()
            self.mar_vals.popleft()
            self.angle_vals.popleft()
            self.blinks.popleft()
            self.yawns.popleft()
            self.emotions.popleft()
            self.hrs.popleft()

    def detect_nod(self, now):
        if len(self.angle_history) >= 2:
            diff = abs(self.angle_history[-1] - self.angle_history[-2])
            if diff > NOD_DIFF_THRESHOLD_DEG and now - self.nod_last_time > NOD_COOLDOWN:
                self.nod_count += 1
                self.nod_last_time = now

    def compute_fatigue(self, baseline_ear):
        if not self.ear_vals:
            return 0.0
        ear_mean = float(np.mean(self.ear_vals))
        ear_score = scaled_score(ear_mean, baseline_ear, scale=22.0)
        mar_mean = float(np.mean(self.mar_vals)) if self.mar_vals else 0.0
        mar_score = float(sigmoid(20.0 * (mar_mean - YAWN_MAR_THRESHOLD)))
        emotion_score = float(np.mean(self.emotions)) if self.emotions else 0.0
        hr_mean = float(np.mean(self.hrs)) if self.hrs else 72.0
        hr_score = float(sigmoid(0.18 * (60 - hr_mean)))
        nod_score = min(1.0, self.nod_count / 5.0)
        weights = {"ear": 0.4, "mar": 0.18, "emotion": 0.12, "hr": 0.10, "nod": 0.20}
        fused = (weights["ear"] * ear_score +
                 weights["mar"] * mar_score +
                 weights["emotion"] * emotion_score +
                 weights["hr"] * hr_score +
                 weights["nod"] * nod_score)
        return float(np.clip(fused, 0.0, 1.0))

# -------------- Calibration ----------------
def calibrate(cap, detector, predictor, duration=CALIBRATE_SECONDS):
    print(f"[INFO] Calibration: look straight ahead for {duration} seconds...")
    ear_vals = []
    angle_vals = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            continue
        shape = predictor(gray, faces[0])
        pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        left_eye = [pts[i] for i in range(36, 42)]
        right_eye = [pts[i] for i in range(42, 48)]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        angle = abs(head_angle(pts[0], pts[16]))
        ear_vals.append(ear)
        angle_vals.append(angle)
    baseline_ear = float(np.mean(ear_vals)) if ear_vals else 0.28
    baseline_angle = float(np.mean(angle_vals)) if angle_vals else 0.0
    print(f"[INFO] Baseline EAR={baseline_ear:.3f}, angle={baseline_angle:.2f}")
    return baseline_ear, baseline_angle

# --------------- Main --------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not available.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("[ERROR] Put shape_predictor_68_face_landmarks.dat in same folder.")
        cap.release()
        return
    predictor = dlib.shape_predictor(predictor_path)

    baseline_ear, baseline_angle = calibrate(cap, detector, predictor)
    ear_threshold = min(DEFAULT_EAR_THRESHOLD, baseline_ear * 0.85)

    tracker = BehaviorTracker(window_secs=ROLLING_WINDOW)
    last_alert_time = 0.0
    last_escalation_time = 0.0
    escalation_stage = 0  # 0 = none, 1 = voice, 2 = alarm, 3 = loud alarm & vehicle suggestion

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if not faces:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("AWARE Drive", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            for face in faces:
                shape = predictor(gray, face)
                pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                left_eye = [pts[i] for i in range(36, 42)]
                right_eye = [pts[i] for i in range(42, 48)]
                mouth = [pts[i] for i in range(48, 68)]
                lp, rp = pts[0], pts[16]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)
                angle = abs(head_angle(lp, rp))
                emotion_score = detect_emotion(frame, (face.left(), face.top(), face.width(), face.height()))
                hr = get_heart_rate()

                blink = 1 if ear < ear_threshold else 0
                yawn = 1 if mar > YAWN_MAR_THRESHOLD else 0

                # detect prolonged eye closure -> possible sleep/pretend
                if ear < ear_threshold:
                    if tracker.eye_closed_since is None:
                        tracker.eye_closed_since = timestamp
                    closed_duration = timestamp - tracker.eye_closed_since
                else:
                    closed_duration = 0.0
                    tracker.eye_closed_since = None

                tracker.add(timestamp, ear, mar, angle, blink, yawn, emotion_score, hr)

                lighting = check_lighting(frame)
                fatigue = tracker.compute_fatigue(baseline_ear)

                # display overlays
                cv2.putText(frame, f"FAT:{fatigue:.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if fatigue < 0.6 else (0,0,255), 2)
                cv2.putText(frame, f"EAR:{ear:.3f}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(frame, f"MAR:{mar:.3f}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(frame, f"HR:{hr}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(frame, f"NOD:{tracker.nod_count}", (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(frame, f"CLOSED:{closed_duration:.2f}s", (10,125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.putText(frame, f"LIGHT:{lighting:.2f}", (10,145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

                # draw landmarks
                for (x,y) in pts:
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)

                now = timestamp

                # Condition: pretending-to-sleep or real-sleep detection:
                sleeping = closed_duration >= PRETEND_SLEEP_SECONDS
                severe_fatigue = fatigue > 0.75 or sleeping

                # Escalating alerts:
                if severe_fatigue:
                    # stage 1: voice warning (if not recently spoken)
                    if escalation_stage < 1 and now - last_alert_time > ALERT_COOLDOWN:
                        async_speak("Warning. You seem drowsy. Please wake up or pull over.")
                        escalation_stage = 1
                        last_alert_time = now
                    # stage 2: gentle alarm sound
                    elif escalation_stage == 1 and now - last_alert_time > ESCALATION_COOLDOWN:
                        if os.path.exists(ALARM_PATH):
                            async_play_sound(ALARM_PATH)
                        else:
                            async_speak("Alarm.wav")
                        escalation_stage = 2
                        last_escalation_time = now
                    # stage 3: loud alarm + vehicle control suggestion (if still severe)
                    elif escalation_stage >= 2 and now - last_escalation_time > ESCALATION_COOLDOWN:
                        if os.path.exists(LOUD_ALARM_PATH):
                            # repeat loud alarm several times
                            for _ in range(2):
                                async_play_sound(LOUD_ALARM_PATH)
                                time.sleep(0.3)
                        else:
                            async_speak("Severe drowsiness detected. Please stop the vehicle immediately.")
                        # vehicle control suggestion (placeholder - implement CAN/OBD integration)
                        suggestion = ("SUGGESTION: Reduce speed, enable hazard lights, "
                                      "pull over to safe location. Notify emergency contact.")
                        print("=== VEHICLE SUGGESTION ===")
                        print(suggestion)
                        escalation_stage = 3
                        last_escalation_time = now
                else:
                    # Reset escalation if user becomes alert again
                    escalation_stage = 0

            cv2.imshow("AWARE Drive", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # save logs
        if tracker.log:
            keys = tracker.log[0].keys()
            try:
                with open(LOG_CSV, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(tracker.log)
                print(f"[INFO] Log saved: {LOG_CSV} ({len(tracker.log)} rows)")
            except Exception as e:
                print("[WARN] Could not save log:", e)

if __name__ == "__main__":
    main()
