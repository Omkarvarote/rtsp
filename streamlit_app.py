"""
RTSP Dual Stream Monitor — Streamlit App
Works with: Webcam, RTSP URL, Video File
Strategy: cv2 reads frames -> saves to temp files -> Streamlit displays them
"""

import cv2
import numpy as np
import threading
import time
import os
import tempfile
import logging
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Stream Monitor", page_icon="📹", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0f1117 !important; color: #e2e8f0 !important; }
.stApp { background: #0f1117 !important; }
section[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2a3a !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
.stButton > button { background: #1e2a3a !important; color: #7dd3fc !important; border: 1px solid #2d3f55 !important; border-radius: 8px !important; font-weight: 500 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; color: white !important; border: none !important; }
pre, code { background: #0a0f18 !important; border: 1px solid #1e2a3a !important; color: #4ade80 !important; border-radius: 8px !important; }
hr { border-color: #1e2a3a !important; }
</style>
""", unsafe_allow_html=True)

MOTION_THRESHOLD     = 500
RECORD_ON_MOTION     = True
RECORD_SECONDS_AFTER = 10
SAVE_DIR             = "recordings"
LOG_FILE             = "monitor.log"
W, H                 = 640, 360
FRAME_DIR            = "stream_frames"  # temp folder for sharing frames

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    encoding="utf-8"
)
log = logging.getLogger("StreamMonitor")


class CameraWorker:
    """
    Runs in a background thread.
    Reads frames from any source (webcam int, RTSP string, file string)
    Saves latest annotated frame as a JPEG file for Streamlit to display.
    """
    def __init__(self, source, name, cam_id):
        self.source  = source
        self.name    = name
        self.cam_id  = cam_id
        self.running = False
        self.frame_path = os.path.join(FRAME_DIR, f"cam{cam_id}.jpg")

        self.bg_sub       = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.motion       = False
        self.last_motion  = 0
        self.writer       = None
        self.recording    = False
        self.record_path  = ""
        self.status       = "Connecting..."
        self.motion_flag  = False

    def open_cap(self):
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.writer:
            self.writer.release()

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.dilate(self.bg_sub.apply(gray), None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > MOTION_THRESHOLD:
                self.last_motion = time.time()
                if not self.motion:
                    self.motion = True
                    log.warning(f"[{self.name}] Motion detected")
                return True
        if self.motion and time.time() - self.last_motion > RECORD_SECONDS_AFTER:
            self.motion = False
            log.info(f"[{self.name}] Motion cleared")
        return self.motion

    def _write_frame(self, frame):
        if not RECORD_ON_MOTION: return
        if self.motion:
            if not self.recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.record_path = os.path.join(SAVE_DIR, f"cam{self.cam_id}_{ts}.avi")
                h, w = frame.shape[:2]
                self.writer    = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w, h))
                self.recording = True
                log.info(f"[{self.name}] Recording -> {self.record_path}")
            if self.writer: self.writer.write(frame)
        elif self.recording:
            self.writer.release(); self.writer = None
            self.recording = False
            log.info(f"[{self.name}] Saved -> {self.record_path}")

    def _annotate(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,40), (15,15,20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, self.name, (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230,230,230), 2)
        ts = datetime.now().strftime("%H:%M:%S  %Y-%m-%d")
        bot = frame.copy()
        cv2.rectangle(bot, (0,h-28),(w,h),(15,15,20),-1)
        cv2.addWeighted(bot, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, ts, (10,h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,200,120), 1)
        if self.motion:
            cv2.rectangle(frame, (0,0),(w-1,h-1),(30,30,230),4)
            lbl = "!! MOTION !!"
            tw  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            cv2.putText(frame, lbl, ((w-tw)//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,255), 2)
        if self.recording:
            cv2.circle(frame, (w-20,20), 8, (50,50,240), -1)
            cv2.putText(frame, "REC", (w-66,26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,255), 1)

    def _loop(self):
        cap = self.open_cap()
        while self.running:
            if not cap.isOpened():
                self.status = "No Signal"
                # Write no-signal frame
                f = np.full((H, W, 3), 14, dtype=np.uint8)
                cv2.putText(f, self.name,     (14,32),      cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55,65,81), 2)
                cv2.putText(f, "No signal",   (W//2-80,H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40,50,65), 2)
                cv2.imwrite(self.frame_path, f)
                cap.release()
                time.sleep(2)
                cap = self.open_cap()
                continue

            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.lower().startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cap.release()
                cap = self.open_cap()
                continue

            self.status = "Live"
            frame = cv2.resize(frame, (W, H))
            self._detect_motion(frame)
            self._write_frame(frame)
            self._annotate(frame)

            # Save frame to disk so Streamlit can read it
            cv2.imwrite(self.frame_path, frame)
            time.sleep(0.033)

        cap.release()


def source_picker(title, prefix):
    st.markdown(f"**{title}**")
    name = st.text_input("Label", value=title, key=f"{prefix}_n")
    mode = st.radio("Source", ["📁 Video File", "📡 RTSP URL", "💻 Webcam"],
                    key=f"{prefix}_m", horizontal=True)
    src = None
    if mode == "📁 Video File":
        up = st.file_uploader("Upload video", type=["mp4","avi","mov"], key=f"{prefix}_u")
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
            tmp.write(up.read()); tmp.flush()
            src = tmp.name
            st.caption(f"✅ {up.name}")
        else:
            st.caption("Upload a `.mp4` or `.avi` file")
    elif mode == "📡 RTSP URL":
        src = st.text_input("URL", value="rtsp://admin:password@192.168.1.100:554/stream1", key=f"{prefix}_r")
        st.caption("e.g. `rtsp://admin:pass@IP:554/stream1`")
    else:
        idx = st.number_input("Webcam Index", 0, 4, 0, key=f"{prefix}_i")
        src = int(idx)
        st.caption("0 = built-in webcam, 1 = USB webcam")
    return src, name


def status_badge(worker, name):
    if worker.motion:
        b = f"<span style='background:#1e1b4b;color:#818cf8;padding:3px 10px;border-radius:20px;font-size:0.78rem;border:1px solid #312e81;'>⬤ Motion – {name}</span>"
    else:
        b = f"<span style='background:#052e16;color:#4ade80;padding:3px 10px;border-radius:20px;font-size:0.78rem;border:1px solid #14532d;'>⬤ Clear – {name}</span>"
    if worker.recording:
        b += " <span style='background:#1e1b4b;color:#a5b4fc;padding:3px 8px;border-radius:20px;font-size:0.75rem;border:1px solid #3730a3;'>⏺ Rec</span>"
    return b


def main():
    st.markdown("""
    <div style="padding:16px 0 4px 0;display:flex;align-items:center;gap:12px;">
        <span style="font-size:1.6rem;">📹</span>
        <div>
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9;">Stream Monitor</div>
            <div style="font-size:0.75rem;color:#475569;">Dual Stream • Motion Detection • Auto Recording</div>
        </div>
    </div>
    <hr style="border-color:#1e2a3a;margin:10px 0 18px 0;">
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        st.markdown("---")
        with st.expander("📷 Camera 1", expanded=True):
            src1, name1 = source_picker("Camera 1", "c1")
        st.markdown(" ")
        with st.expander("📷 Camera 2", expanded=True):
            src2, name2 = source_picker("Camera 2", "c2")
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            start = st.button("▶ Start", type="primary", use_container_width=True)
        with col_b:
            stop = st.button("⏹ Stop", use_container_width=True)
        st.caption(f"💾 `{SAVE_DIR}/`  |  📋 `{LOG_FILE}`")

    # ── Stop ──
    if stop:
        if "w1" in st.session_state: st.session_state["w1"].stop()
        if "w2" in st.session_state: st.session_state["w2"].stop()
        for k in ["w1","w2","src1","src2","name1","name2","running"]:
            st.session_state.pop(k, None)
        st.info("Streams stopped.")
        st.stop()

    # ── Start ──
    if start:
        if src1 is None or src2 is None:
            st.error("Please configure both cameras first.")
            st.stop()
        if "w1" in st.session_state: st.session_state["w1"].stop()
        if "w2" in st.session_state: st.session_state["w2"].stop()

        w1 = CameraWorker(src1, name1, 1)
        w2 = CameraWorker(src2, name2, 2)
        w1.start(); w2.start()

        st.session_state["w1"]      = w1
        st.session_state["w2"]      = w2
        st.session_state["name1"]   = name1
        st.session_state["name2"]   = name2
        st.session_state["running"] = True

    # ── Welcome ──
    if "running" not in st.session_state:
        st.markdown("""
        <div style="background:#161b27;border:1px solid #1e2a3a;border-radius:14px;padding:40px 32px;text-align:center;margin-top:20px;">
            <div style="font-size:2.5rem;margin-bottom:12px;">📹</div>
            <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-bottom:8px;">Ready to monitor</div>
            <div style="color:#475569;font-size:0.85rem;margin-bottom:28px;">Configure cameras in sidebar → press ▶ Start</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;text-align:left;">
                <div style="background:#0f1117;border:1px solid #1e2a3a;border-radius:10px;padding:18px;">
                    <div style="font-size:1.3rem;">📁</div>
                    <div style="font-weight:600;color:#7dd3fc;font-size:0.85rem;margin-top:6px;">Upload Video</div>
                    <div style="color:#475569;font-size:0.78rem;margin-top:4px;">Any .mp4 or .avi file</div>
                </div>
                <div style="background:#0f1117;border:1px solid #1e2a3a;border-radius:10px;padding:18px;">
                    <div style="font-size:1.3rem;">📡</div>
                    <div style="font-weight:600;color:#7dd3fc;font-size:0.85rem;margin-top:6px;">RTSP Camera</div>
                    <div style="color:#475569;font-size:0.78rem;margin-top:4px;">Real IP camera URL</div>
                </div>
                <div style="background:#0f1117;border:1px solid #1e2a3a;border-radius:10px;padding:18px;">
                    <div style="font-size:1.3rem;">💻</div>
                    <div style="font-weight:600;color:#7dd3fc;font-size:0.85rem;margin-top:6px;">Webcam</div>
                    <div style="color:#475569;font-size:0.78rem;margin-top:4px;">Index 0 = built-in cam</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    w1    = st.session_state["w1"]
    w2    = st.session_state["w2"]
    name1 = st.session_state["name1"]
    name2 = st.session_state["name2"]

    col1, col2 = st.columns(2)
    with col1:
        ph1    = st.empty()
        badge1 = st.empty()
    with col2:
        ph2    = st.empty()
        badge2 = st.empty()

    st.markdown("<hr style='border-color:#1e2a3a;margin:14px 0;'>", unsafe_allow_html=True)
    lc, rc = st.columns([3, 1])
    with lc:
        st.markdown("##### 📋 Event Log")
        log_ph = st.empty()
    with rc:
        st.markdown("##### 🎥 Recordings")
        rec_ph = st.empty()

    # ── Display loop — reads saved JPEG files ──
    while True:
        # Camera 1
        if os.path.exists(w1.frame_path):
            img1 = cv2.imread(w1.frame_path)
            if img1 is not None:
                ph1.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            ns = np.full((H,W,3),14,dtype=np.uint8)
            cv2.putText(ns,"No signal",(W//2-80,H//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(40,50,65),2)
            ph1.image(cv2.cvtColor(ns,cv2.COLOR_BGR2RGB), use_container_width=True)

        # Camera 2
        if os.path.exists(w2.frame_path):
            img2 = cv2.imread(w2.frame_path)
            if img2 is not None:
                ph2.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            ns = np.full((H,W,3),14,dtype=np.uint8)
            cv2.putText(ns,"No signal",(W//2-80,H//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(40,50,65),2)
            ph2.image(cv2.cvtColor(ns,cv2.COLOR_BGR2RGB), use_container_width=True)

        badge1.markdown(status_badge(w1, name1), unsafe_allow_html=True)
        badge2.markdown(status_badge(w2, name2), unsafe_allow_html=True)

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            log_ph.code("".join(lines[-16:]), language=None)
        except Exception:
            log_ph.caption("No events yet.")

        recs = sorted(os.listdir(SAVE_DIR), reverse=True) if os.path.exists(SAVE_DIR) else []
        if recs:
            rec_ph.dataframe({"File": recs[:8]}, use_container_width=True, hide_index=True)
        else:
            rec_ph.caption("No recordings yet.")

        time.sleep(0.033)


if __name__ == "__main__":
    main()