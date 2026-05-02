# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:46:53 2026

@author: durga
"""

import cv2
import time
import threading
from ultralytics import YOLO
from playsound import playsound
from twilio.rest import Client

# ================== TWILIO SETUP ==================
account_sid = "AC1be0fd0aac879f14dd4995bcca4ae48f"
auth_token = "9171bc43110e2db456f7c4796da5f918"

client = Client(account_sid, auth_token)

from_number = "+13614410596"      # your Twilio number
to_number = "+919059081597"       # your phone number


def send_sms(text):
    try:
        client.messages.create(
            body=text,
            from_=from_number,
            to=to_number
        )
        print("SMS sent:", text)
    except Exception as e:
        print("SMS error:", e)


# ================== MODEL ==================
model = YOLO("C:/Users/durga/AI_Project/best.pt")

cap = cv2.VideoCapture(0)

# ================== CONTROL ==================
alarm_on = False
last_label = None
last_time = 0
cooldown = 10   # seconds


def play_sound(path):
    global alarm_on
    playsound(path)
    alarm_on = False


# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        print(label, round(conf, 2))   # debug

        # ================== FILTER ==================
        if conf < 0.5:
            continue

        # stricter for bird (reduces false detection)
        if label == "bird" and conf < 0.65:
            continue

        # ================== BOX ==================
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "monkey":
            color = (0, 255, 0)
            sound = r"C:\Users\durga\AI_PROJECT\monkey.mp3.mpeg"

        elif label == "elephant":
            color = (255, 0, 0)
            sound = r"C:\Users\durga\AI_PROJECT\elephant.mp3.mpeg"

        elif label == "bird":
            color = (0, 0, 255)
            sound = r"C:\Users\durga\AI_PROJECT\bird.mp3.mpeg"

        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        # ================== ALERT ==================
        current_time = time.time()

        if label != last_label or current_time - last_time > cooldown:

            # SOUND
            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_sound, args=(sound,)).start()

            # SMS
            send_sms(f"{label.upper()} detected!")

            last_label = label
            last_time = current_time

    cv2.imshow("Smart Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
