import streamlit as st
import cv2
import numpy as np
import threading
import time
import os
import tempfile
from pathlib import Path

# ─── Try importing optional libs ──────────────────────────────────────────────
try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Chidori - 1000 Birds",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #0a0a0a; color: white; }
    .stApp { background-color: #0a0a0a; }
    h1 { color: #00ffff; text-shadow: 0 0 20px #00ffff; }
    .chidori-title {
        font-size: 3rem;
        font-weight: bold;
        color: #00bfff;
        text-align: center;
        text-shadow: 0 0 30px #00bfff, 0 0 60px #0080ff;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chidori-title">⚡ CHIDORI ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">千鳥 — One Thousand Birds</div>', unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "chidori_active" not in st.session_state:
    st.session_state.chidori_active = False
if "sound_played" not in st.session_state:
    st.session_state.sound_played = False

# ─── Sidebar — file upload ──────────────────────────────────────────────────────
st.sidebar.title("⚙️ Setup Files")
st.sidebar.markdown("Upload your Chidori assets:")

sound_file = st.sidebar.file_uploader(
    "🎵 Chidori Sound (MP3/WAV)",
    type=["mp3", "wav"],
    help="Upload your chidori_sound.mp3"
)

video_file = st.sidebar.file_uploader(
    "🎬 Chidori Effect Video (MP4)",
    type=["mp4", "avi"],
    help="Upload your chidori_effect.mp4 (green screen)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Controls:**")
activate_mode = st.sidebar.radio(
    "Activation Mode",
    ["👐 Show Hand (Auto)", "🎙️ Say 'Chidori' (Voice)"],
    index=0
)

sensitivity = st.sidebar.slider("Hand Detection Sensitivity", 3000, 15000, 5000, 500)

# ─── Save uploaded files to temp ───────────────────────────────────────────────
sound_path = None
video_path = None

if sound_file:
    tmp_sound = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_sound.write(sound_file.read())
    tmp_sound.close()
    sound_path = tmp_sound.name
    st.sidebar.success("✅ Sound loaded!")

if video_file:
    tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_vid.write(video_file.read())
    tmp_vid.close()
    video_path = tmp_vid.name
    st.sidebar.success("✅ Video loaded!")

# ─── Sound player ──────────────────────────────────────────────────────────────
def play_chidori_sound(path):
    """Play the chidori sound effect using pygame."""
    if not PYGAME_OK or not path:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Sound error: {e}")

# ─── Green screen removal ───────────────────────────────────────────────────────
def remove_green_screen(effect_frame):
    """
    Remove green background from effect frame.
    Returns (effect_rgb, mask) where mask is the alpha channel.
    """
    hsv = cv2.cvtColor(effect_frame, cv2.COLOR_BGR2HSV)

    # Green screen range — tune if your video has a different green shade
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Also try removing pure black (some green screen vids have black bg)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    combined_mask = cv2.bitwise_or(green_mask, black_mask)

    # Invert: 255 = keep effect pixels, 0 = transparent
    alpha_mask = cv2.bitwise_not(combined_mask)

    # Clean up edges
    kernel = np.ones((3, 3), np.uint8)
    alpha_mask = cv2.erode(alpha_mask, kernel, iterations=1)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0)

    return effect_frame, alpha_mask

# ─── Overlay effect on hand ─────────────────────────────────────────────────────
def overlay_effect(background, effect_frame, cx, cy, size=200):
    """
    Overlay the chidori effect (with green screen removed) onto the background
    centered at (cx, cy).
    """
    effect_rgb, alpha_mask = remove_green_screen(effect_frame)

    # Resize effect to desired size
    effect_resized = cv2.resize(effect_rgb, (size, size))
    alpha_resized = cv2.resize(alpha_mask, (size, size))

    h_bg, w_bg = background.shape[:2]

    # Calculate overlay region
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w_bg, x1 + size)
    y2 = min(h_bg, y1 + size)

    # Crop effect to fit within frame bounds
    ex1 = x1 - (cx - size // 2)
    ey1 = y1 - (cy - size // 2)
    ex2 = ex1 + (x2 - x1)
    ey2 = ey1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return background

    # Extract regions
    bg_region = background[y1:y2, x1:x2].astype(float)
    ef_region = effect_resized[ey1:ey2, ex1:ex2].astype(float)
    al_region = alpha_resized[ey1:ey2, ex1:ex2].astype(float) / 255.0

    # Blend
    al_3ch = np.stack([al_region] * 3, axis=-1)
    blended = ef_region * al_3ch + bg_region * (1 - al_3ch)
    background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return background

# ─── Lightning particle effect (fallback if no video) ──────────────────────────
def draw_lightning_effect(frame, cx, cy):
    """Draw a procedural blue lightning effect around hand center."""
    output = frame.copy()

    # Outer electric glow rings
    for r, alpha in [(80, 0.15), (60, 0.25), (40, 0.4)]:
        overlay = output.copy()
        cv2.circle(overlay, (cx, cy), r, (255, 200, 50), 2)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Inner bright core
    for r, alpha in [(25, 0.6), (15, 0.8), (8, 1.0)]:
        cv2.circle(output, (cx, cy), r, (255, 255, 180), -1)
        overlay = output.copy()
        cv2.addWeighted(overlay, alpha * 0.3, output, 1 - alpha * 0.3, 0, output)

    cv2.circle(output, (cx, cy), 8, (200, 230, 255), -1)

    # Random lightning bolts
    num_bolts = 8
    for i in range(num_bolts):
        angle = (2 * np.pi / num_bolts) * i + np.random.uniform(-0.3, 0.3)
        length = np.random.randint(40, 90)

        pts = [(cx, cy)]
        x, y = float(cx), float(cy)
        dx = np.cos(angle) * (length / 4)
        dy = np.sin(angle) * (length / 4)

        for seg in range(4):
            jitter_x = np.random.uniform(-12, 12)
            jitter_y = np.random.uniform(-12, 12)
            x += dx + jitter_x
            y += dy + jitter_y
            pts.append((int(x), int(y)))

        # Draw thick glow then thin bright bolt
        for j in range(len(pts) - 1):
            cv2.line(output, pts[j], pts[j + 1], (100, 150, 255), 3)
        for j in range(len(pts) - 1):
            cv2.line(output, pts[j], pts[j + 1], (220, 240, 255), 1)

    # Electric blue tint overlay on whole frame
    blue_overlay = np.zeros_like(frame)
    blue_overlay[:, :] = (20, 10, 0)  # BGR: slight blue tint
    cv2.addWeighted(blue_overlay, 0.1, output, 0.9, 0, output)

    return output

# ─── Voice listener thread ──────────────────────────────────────────────────────
voice_triggered = threading.Event()

def listen_for_chidori():
    """Background thread: listen for the word 'chidori'."""
    if not SR_OK:
        return
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while st.session_state.running:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            if "chidori" in text or "chidori" in text:
                voice_triggered.set()
        except Exception:
            pass

# ─── Main camera loop ───────────────────────────────────────────────────────────
def run_camera(frame_placeholder, status_placeholder, sound_path, video_path,
               activate_mode, sensitivity):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("❌ Cannot open webcam. Make sure it's connected!")
        return

    # Load chidori effect video
    effect_cap = None
    if video_path:
        effect_cap = cv2.VideoCapture(video_path)

    chidori_active = False
    chidori_start_time = 0
    chidori_duration = 4.0   # seconds to show effect
    sound_played = False
    effect_frame_idx = 0

    # Voice thread
    if "Voice" in activate_mode and SR_OK:
        voice_thread = threading.Thread(target=listen_for_chidori, daemon=True)
        voice_thread.start()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ── Hand Detection ──────────────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hand_detected = False
        cx, cy = w // 2, h // 2

        if contours:
            hand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand) > sensitivity:
                hand_detected = True

                M = cv2.moments(hand)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                # Draw hand outline
                cv2.drawContours(frame, [hand], -1, (0, 255, 100), 2)
                cv2.circle(frame, (cx, cy), 8, (0, 100, 255), -1)

        # ── Activation Logic ────────────────────────────────────────────────
        should_activate = False

        if "Auto" in activate_mode and hand_detected:
            should_activate = True
        elif "Voice" in activate_mode and voice_triggered.is_set():
            should_activate = True
            voice_triggered.clear()

        if should_activate and not chidori_active:
            chidori_active = True
            chidori_start_time = time.time()
            sound_played = False

        # ── Chidori Effect ──────────────────────────────────────────────────
        if chidori_active:
            elapsed = time.time() - chidori_start_time

            if elapsed > chidori_duration:
                chidori_active = False
                sound_played = False
                if effect_cap:
                    effect_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # Play sound once at activation
                if not sound_played and sound_path:
                    play_chidori_sound(sound_path)
                    sound_played = True

                # Overlay video effect or fallback lightning
                if effect_cap and effect_cap.isOpened():
                    ret_e, effect_frame = effect_cap.read()
                    if not ret_e:
                        effect_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_e, effect_frame = effect_cap.read()

                    if ret_e:
                        frame = overlay_effect(frame, effect_frame, cx, cy, size=220)
                else:
                    frame = draw_lightning_effect(frame, cx, cy)

                # Chidori text with glow style
                pulse = int(abs(np.sin(elapsed * 6)) * 30)
                color = (255, 200 + pulse, 50 + pulse)
                cv2.putText(frame, "CHIDORI!",
                            (w // 2 - 100, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, (100, 150, 255), 6)
                cv2.putText(frame, "CHIDORI!",
                            (w // 2 - 100, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2)

                cv2.putText(frame, "* 1000 BIRDS *",
                            (w // 2 - 90, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 1)
        else:
            # Status text when idle
            if hand_detected:
                cv2.putText(frame, "Hand Detected - Activating...",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 150), 2)
            else:
                cv2.putText(frame, "Show your hand to the camera",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (150, 150, 150), 1)

        # ── Convert BGR to RGB for Streamlit ───────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Status
        if chidori_active:
            status_placeholder.markdown("### ⚡ CHIDORI ACTIVE — 千鳥発動！")
        elif hand_detected:
            status_placeholder.markdown("### 👐 Hand detected!")
        else:
            status_placeholder.markdown("### 👁️ Waiting for hand...")

        time.sleep(0.033)  # ~30 FPS

    cap.release()
    if effect_cap:
        effect_cap.release()
    frame_placeholder.empty()
    status_placeholder.markdown("### ⏹️ Camera stopped.")

# ─── UI Layout ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    start_btn = st.button("▶️ Start Camera", type="primary", use_container_width=True)

with col2:
    stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

with col3:
    manual_btn = st.button("⚡ Trigger Chidori!", use_container_width=True)

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

# ─── Camera frame display ───────────────────────────────────────────────────────
frame_placeholder = st.empty()
status_placeholder = st.empty()

if st.session_state.running:
    if not sound_file and not video_file:
        st.warning("⚠️ No files uploaded! Chidori will use the built-in lightning effect and no sound. Upload files in the sidebar for the full experience!")

    run_camera(
        frame_placeholder,
        status_placeholder,
        sound_path,
        video_path,
        activate_mode,
        sensitivity
    )

# ─── Instructions ───────────────────────────────────────────────────────────────
with st.expander("📖 How to use"):
    st.markdown("""
    **Setup:**
    1. Upload your `chidori_sound.mp3` and `chidori_effect.mp4` in the sidebar
    2. Choose activation mode: **Auto** (show hand) or **Voice** (say "Chidori!")
    3. Click **▶️ Start Camera**

    **Activation Modes:**
    - 👐 **Show Hand** — Chidori fires automatically when your hand is detected
    - 🎙️ **Say 'Chidori'** — Say the word out loud to trigger it

    **Tips:**
    - Make sure your room is well lit for better hand detection
    - Adjust the **sensitivity slider** if detection is too sensitive/not sensitive enough
    - The effect video needs a **green screen** background for best results
    - If no video is uploaded, a cool built-in lightning effect will show instead!

    **Deployment:**
    - Run locally: `streamlit run app.py`
    - Deploy to Streamlit Cloud: Push to GitHub, then connect at share.streamlit.io
    """)

with st.expander("📦 Requirements"):
    st.code("""
# requirements.txt
streamlit
opencv-python-headless
numpy
pygame
SpeechRecognition
pyaudio
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444'>⚡ Built with Sasuke energy ⚡</div>",
    unsafe_allow_html=True
)
