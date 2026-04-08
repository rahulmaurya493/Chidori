import cv2
import numpy as np
import streamlit as st
from PIL import Image
import base64

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Chidori - Naruto",
    page_icon="⚡",
    layout="centered"
)

# ─── Custom Dark Styling ────────────────────────────────────
st.markdown("""
    <style>
    body { background-color: #0a0a0a; }
    .main { background-color: #0a0a0a; }
    h1 { color: #00cfff; text-align: center; }
    .stButton>button {
        background-color: #00cfff;
        color: black;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        width: 100%;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Title ──────────────────────────────────────────────────
st.markdown("<h1>⚡ CHIDORI - 1000 BIRDS ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Show your hand → Click Activate → Feel the power!</p>", unsafe_allow_html=True)

# ─── Load Audio ─────────────────────────────────────────────
def get_audio_base64(audio_path):
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

# ─── Load GIF frames ────────────────────────────────────────
def load_gif_frames(gif_path):
    try:
        cap = cv2.VideoCapture(gif_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    except:
        return []

# ─── Skin Detection ─────────────────────────────────────────
def detect_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) > 3000:
            M = cv2.moments(hand)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), hand
    return None, None

# ─── Overlay Chidori Effect ──────────────────────────────────
def overlay_chidori(frame, center, effect_frame):
    h, w, _ = frame.shape
    effect = cv2.resize(effect_frame, (220, 220))
    ex = max(0, center[0] - 110)
    ey = max(0, center[1] - 110)
    eh, ew, _ = effect.shape

    if ey + eh <= h and ex + ew <= w:
        roi = frame[ey:ey+eh, ex:ex+ew]
        blended = cv2.addWeighted(roi, 0.3, effect, 0.7, 0)
        frame[ey:ey+eh, ex:ex+ew] = blended

    return frame

# ─── Session State ───────────────────────────────────────────
if "chidori_active" not in st.session_state:
    st.session_state.chidori_active = False
if "effect_frame_idx" not in st.session_state:
    st.session_state.effect_frame_idx = 0

# ─── Camera Input ────────────────────────────────────────────
camera_input = st.camera_input("📸 Show your hand here!")

# ─── Activate Button ─────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if st.button("⚡ ACTIVATE CHIDORI!"):
        st.session_state.chidori_active = True
        st.session_state.effect_frame_idx = 0
with col2:
    if st.button("❌ Deactivate"):
        st.session_state.chidori_active = False
        st.session_state.effect_frame_idx = 0

# ─── Process Frame ───────────────────────────────────────────
if camera_input is not None:
    bytes_data = camera_input.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    # Detect hand
    center, hand_contour = detect_hand(frame)

    if center:
        # Draw hand outline
        if hand_contour is not None:
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, center, 10, (0, 0, 255), -1)

        # Chidori effect active
        if st.session_state.chidori_active:
            gif_frames = load_gif_frames("chidori_effect.gif")

            if gif_frames:
                idx = st.session_state.effect_frame_idx % len(gif_frames)
                frame = overlay_chidori(frame, center, gif_frames[idx])
                st.session_state.effect_frame_idx += 1

            # Lightning text
            cv2.putText(frame, "⚡ CHIDORI!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2,
                       (255, 255, 0), 3)

        st.success("✅ Hand Detected!")
    else:
        st.warning("🖐️ Show your hand clearly in good lighting!")

    # Show final frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption="⚡ Chidori Effect", use_column_width=True)

# ─── Play Sound ──────────────────────────────────────────────
if st.session_state.chidori_active:
    audio_b64 = get_audio_base64("chidori_sound.mp3")
    if audio_b64:
        st.markdown(f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """, unsafe_allow_html=True)
    st.markdown("""
        <h2 style='text-align:center; color:#00cfff;'>
        ⚡⚡ CHIDORI ACTIVATED! ⚡⚡
        </h2>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built by Rahul ⚡ | Naruto x AI</p>", unsafe_allow_html=True)
