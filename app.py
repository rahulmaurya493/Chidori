import cv2
import numpy as np
import pygame
import speech_recognition as sr
import threading

# Initialize sound
pygame.mixer.init()
pygame.mixer.music.load("chidori_sound.mp3")

# Load chidori GIF frames
def load_gif_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

chidori_frames = load_gif_frames("chidori_effect.gif")
frame_index = 0

# State
chidori_active = False

# Voice listener in background thread
def listen_for_chidori():
    global chidori_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Listening for 'Chidori'...")
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=3)
            text = recognizer.recognize_google(audio).lower()
            print(f"Heard: {text}")
            if "chidori" in text:
                chidori_active = True
                pygame.mixer.music.play()
                print("⚡ CHIDORI ACTIVATED!")
        except:
            pass

# Start voice thread
voice_thread = threading.Thread(target=listen_for_chidori, daemon=True)
voice_thread.start()

# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hand_center = None
    if contours:
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) > 5000:
            M = cv2.moments(hand)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hand_center = (cx, cy)
                cv2.drawContours(frame, [hand], -1, (0, 255, 0), 2)

    # Show Chidori effect on hand
    if chidori_active and hand_center and len(chidori_frames) > 0:
        effect = chidori_frames[frame_index % len(chidori_frames)]
        frame_index += 1

        # Resize effect to hand size
        effect = cv2.resize(effect, (200, 200))
        ex, ey = hand_center[0] - 100, hand_center[1] - 100
        ex, ey = max(0, ex), max(0, ey)

        # Overlay effect on frame
        eh, ew, _ = effect.shape
        if ey+eh < h and ex+ew < w:
            frame[ey:ey+eh, ex:ex+ew] = cv2.addWeighted(
                frame[ey:ey+eh, ex:ex+ew], 0.3,
                effect, 0.7, 0
            )

        cv2.putText(frame, "⚡ CHIDORI! ⚡", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)

        # Stop after 3 seconds
        if frame_index > len(chidori_frames) * 2:
            chidori_active = False
            frame_index = 0

    # Status text
    status = "⚡ CHIDORI ACTIVE!" if chidori_active else "🎤 Say 'Chidori' to activate"
    cv2.putText(frame, status, (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("⚡ Chidori - Naruto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
