"""
SMART QUEUELESS COUNTER  (fixed version)
AI Powered Digital Queue Management System

Run:  streamlit run app.py
"""

from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Smart QueueLess Counter",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML HELPER  (FIX #1)
# Markdown turns any line indented by 4+ spaces into a CODE BLOCK.
# That is why your HTML was printed as text. We strip all leading
# whitespace and blank lines so Streamlit always renders it as HTML.
# ============================================================
def html(markup: str) -> None:
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# Newer Streamlit versions deprecate use_container_width
_ST_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}


# ============================================================
# CSS
# ============================================================
html(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124,58,237,0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(6,182,212,0.12), transparent 30%),
        linear-gradient(135deg, #070b19 0%, #0c1224 50%, #10182d 100%);
    color: #f8fafc;
}

/* FIX #2: the white top bar */
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] * { color: #e2e8f0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1d 0%, #10172c 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] label p { color: #f8fafc !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.035);
    border-radius: 12px; padding: 8px 10px; margin-bottom: 4px; transition: 0.2s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(124,58,237,0.20); }

/* Titles */
.main-title {
    font-size: 46px; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(90deg, #c4b5fd, #22d3ee, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.subtitle { color: #cbd5e1; font-size: 17px; margin-bottom: 18px; }
.anime-icons {
    text-align: center; font-size: 55px; padding: 8px; letter-spacing: 10px;
    filter: drop-shadow(0px 8px 20px rgba(124,58,237,0.35));
}

/* Cards */
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.12); border-radius: 22px; padding: 24px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25); backdrop-filter: blur(12px);
    margin-bottom: 18px; color: #f8fafc;
}
.glass-card h3 { color: #ffffff; }
.glass-card p { color: #cbd5e1; }

.metric-card {
    background: linear-gradient(145deg, rgba(124,58,237,0.22), rgba(8,145,178,0.12));
    border: 1px solid rgba(167,139,250,0.25); border-radius: 20px; padding: 20px;
    min-height: 125px; box-shadow: 0 12px 30px rgba(0,0,0,0.20);
}
.metric-number { font-size: 34px; font-weight: 900; color: #ffffff; }
.metric-label { font-size: 14px; color: #cbd5e1; margin-top: 5px; }

.serving-card {
    background: radial-gradient(circle at top, rgba(124,58,237,0.30), rgba(15,23,42,0.95));
    border: 1px solid rgba(167,139,250,0.35); border-radius: 26px;
    padding: 34px 20px; text-align: center; min-height: 280px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35); color: #ffffff;
}
.serving-title { color: #cbd5e1; font-size: 15px; font-weight: 700; letter-spacing: 2px; }
.token-number {
    font-size: 72px; font-weight: 900; margin: 10px 0;
    background: linear-gradient(90deg, #c4b5fd, #67e8f9, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.live-badge {
    display: inline-block; padding: 7px 15px; border-radius: 30px;
    background: rgba(34,197,94,0.15); color: #86efac !important; font-weight: 800;
    border: 1px solid rgba(34,197,94,0.25);
}
.status-box {
    background: rgba(15,23,42,0.75); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px; padding: 24px; color: #f8fafc;
}
.status-box h3 { color: #ffffff; }
.status-box li { color: #cbd5e1; margin: 10px 0; }
.status-box p { color: #cbd5e1; }

.neon-line {
    height: 2px; margin: 22px 0;
    background: linear-gradient(90deg, transparent, #8b5cf6, #22d3ee, #f472b6, transparent);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, #7c3aed, #0891b2);
    color: #ffffff !important; font-weight: 800; min-height: 45px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 30px rgba(124,58,237,0.40);
}
.stButton > button p, .stDownloadButton > button p { color: #ffffff !important; }

/* FIX #3: st.metric text was dark-on-dark (invisible in Analytics) */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 14px 18px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #ffffff !important; font-weight: 800; }

/* Alerts (success / info) readable on dark background */
[data-testid="stAlert"] * { color: #f8fafc !important; }

/* Misc */
input, textarea { border-radius: 12px !important; }
[data-testid="stDataFrame"] { border-radius: 15px; overflow: hidden; }
h1, h2, h3, h4 { color: #ffffff !important; }
.stMarkdown, .stText, .stRadio label, .stSelectbox label { color: #e2e8f0; }
</style>
"""
)


# ============================================================
# DATABASE
# ============================================================
engine = create_engine("sqlite:///queueless.db", connect_args={"check_same_thread": False})
Base = declarative_base()


class Token(Base):
    __tablename__ = "tokens"

    id = Column(Integer, primary_key=True)
    token_number = Column(String)
    service = Column(String)
    status = Column(String)  # waiting | serving | served
    counter = Column(Integer, default=0)
    created_at = Column(DateTime)
    served_at = Column(DateTime, nullable=True)
    wait_minutes = Column(Float, default=0)


Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)


# ============================================================
# SESSION STATE
# ============================================================
if "counters" not in st.session_state:
    st.session_state.counters = {1: True, 2: True, 3: False, 4: False}  # counter -> open?

if "flash" not in st.session_state:
    st.session_state.flash = None


# ============================================================
# HELPERS
# ============================================================
def get_open_counters() -> int:
    return sum(1 for is_open in st.session_state.counters.values() if is_open)


def get_next_token(db) -> str:
    # FIX #4: token numbers now come from the database, so they never repeat
    # after a page refresh / app restart (session_state used to reset to 0).
    last_id = db.query(func.max(Token.id)).scalar() or 0
    return f"A{last_id + 1:02d}"


def get_waiting_tokens():
    db = SessionLocal()
    try:
        return db.query(Token).filter(Token.status == "waiting").order_by(Token.id.asc()).all()
    finally:
        db.close()


def get_current_token() -> str:
    db = SessionLocal()
    try:
        token = (
            db.query(Token).filter(Token.status == "serving").order_by(Token.id.desc()).first()
        )
        return token.token_number if token else "A00"
    finally:
        db.close()


def get_counter_current(counter_number: int) -> str:
    # FIX #5: read from the DB so the Staff Console is always in sync.
    db = SessionLocal()
    try:
        token = (
            db.query(Token)
            .filter(Token.status == "serving", Token.counter == counter_number)
            .first()
        )
        return token.token_number if token else "A00"
    finally:
        db.close()


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 500
    queue_size = rng.integers(1, 80, n)
    counters = rng.integers(1, 5, n)
    service_time = rng.uniform(2, 6, n)
    peak_factor = rng.uniform(0.8, 1.5, n)

    wait_time = queue_size * service_time / counters * peak_factor
    wait_time = wait_time + rng.normal(0, 1.5, n)

    X = np.column_stack((queue_size, counters, service_time, peak_factor))
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, wait_time)
    return model


model = train_model()


def predict_wait_time(queue_size, counters, service_time=3.5) -> float:
    counters = max(counters, 1)
    queue_size = max(queue_size, 0)
    data = np.array([[queue_size, counters, service_time, 1.0]])
    prediction = float(model.predict(data)[0])
    return max(round(prediction, 1), 0)


# ============================================================
# TOKEN ACTIONS
# ============================================================
def create_token(service: str):
    waiting = get_waiting_tokens()
    wait = predict_wait_time(len(waiting) + 1, get_open_counters())

    db = SessionLocal()
    try:
        token_number = get_next_token(db)
        db.add(
            Token(
                token_number=token_number,
                service=service,
                status="waiting",
                counter=0,
                created_at=datetime.now(),
                wait_minutes=wait,
            )
        )
        db.commit()
    finally:
        db.close()
    return token_number, wait


def call_next(counter_number: int):
    db = SessionLocal()
    try:
        current = (
            db.query(Token)
            .filter(Token.status == "serving", Token.counter == counter_number)
            .first()
        )
        if current:
            return current.token_number, True  # already serving someone

        token = db.query(Token).filter(Token.status == "waiting").order_by(Token.id.asc()).first()
        if token is None:
            return None, False

        token.status = "serving"
        token.counter = counter_number
        db.commit()
        return token.token_number, False
    finally:
        db.close()


def complete_token(counter_number: int):
    db = SessionLocal()
    try:
        token = (
            db.query(Token)
            .filter(Token.status == "serving", Token.counter == counter_number)
            .first()
        )
        if token is None:
            return None
        token.status = "served"
        token.served_at = datetime.now()
        db.commit()
        return token.token_number
    finally:
        db.close()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    html(
        """
        <div style="text-align:center; padding:10px 0 20px 0;">
            <div style="font-size:42px;">🎫</div>
            <div style="font-size:23px; font-weight:900; color:#ffffff;">SMART QUEUELESS</div>
            <div style="font-size:12px; color:#94a3b8; margin-top:5px;">AI QUEUE MANAGEMENT</div>
        </div>
        """
    )

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "🎫 Get Token",
            "📷 AI Queue Vision",
            "🧑‍💼 Staff Console",
            "📊 Analytics",
            "⚙️ System",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    html(
        """
        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
                    border-radius:18px; padding:18px;">
            <div style="color:#86efac; font-weight:800; font-size:15px;">🟢 SYSTEM ONLINE</div>
            <br>
            <div style="color:#e2e8f0;">🤖 AI Engine: <b style="color:#67e8f9;">READY</b></div>
            <div style="color:#e2e8f0;">📷 Queue Vision: <b style="color:#67e8f9;">READY</b></div>
            <div style="color:#e2e8f0;">💾 Database: <b style="color:#86efac;">CONNECTED</b></div>
            <div style="color:#e2e8f0;">🏢 Counters: <b style="color:#c4b5fd;">ACTIVE</b></div>
        </div>
        """
    )


# ============================================================
# HEADER
# ============================================================
html('<div class="main-title">🎫 Smart QueueLess Counter</div>')
html(
    '<div class="subtitle">AI-powered digital token system • Smart queue prediction • '
    "Computer vision • Zero unnecessary waiting</div>"
)
html('<div class="anime-icons">🧑‍💻 ✨ 🎫 ✨ 🤖 ✨ 👩‍💻</div>')


# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    waiting = get_waiting_tokens()
    queue_size = len(waiting)
    counters = get_open_counters()
    predicted_wait = predict_wait_time(queue_size, counters)
    current = get_current_token()

    db = SessionLocal()
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        served_today = (
            db.query(Token)
            .filter(Token.status == "served", Token.served_at >= today_start)
            .count()
        )
    finally:
        db.close()

    c1, c2, c3, c4 = st.columns(4)
    for col, number, label in [
        (c1, queue_size, "👥 People Waiting"),
        (c2, f"{predicted_wait:.0f} min", "⏱️ Estimated Wait"),
        (c3, f"{counters}/4", "🏢 Counters Open"),
        (c4, served_today, "✅ Served Today"),
    ]:
        with col:
            html(
                f"""
                <div class="metric-card">
                    <div class="metric-number">{number}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
            )

    html('<div class="neon-line"></div>')

    left, right = st.columns(2)

    with left:
        html(
            f"""
            <div class="serving-card">
                <div class="serving-title">NOW SERVING</div>
                <div class="token-number">{current}</div>
                <div class="live-badge">● LIVE</div>
                <br><br>
                <div style="color:#cbd5e1; font-size:14px;">
                    Please proceed to your assigned counter.
                </div>
            </div>
            """
        )

    with right:
        html(
            """
            <div class="status-box">
                <h3>🧠 Queue Intelligence</h3>
                <p>Smart QueueLess continuously evaluates:</p>
                <ul>
                    <li>👥 <b>Number of people</b></li>
                    <li>🏢 <b>Active counters</b></li>
                    <li>⏱️ <b>Average service time</b></li>
                    <li>📈 <b>Queue growth</b></li>
                    <li>🔥 <b>Peak-hour pressure</b></li>
                </ul>
                <p>The system uses these parameters to estimate the expected waiting time.</p>
            </div>
            """
        )

    st.markdown("## 🎫 Live Queue Board")

    if waiting:
        df = pd.DataFrame(
            [
                {
                    "Position": i + 1,
                    "Token": t.token_number,
                    "Service": t.service,
                    "Estimated Wait": f"{t.wait_minutes:.0f} min",
                    "Status": "🟡 Waiting",
                }
                for i, t in enumerate(waiting)
            ]
        )
        st.dataframe(df, hide_index=True, **STRETCH)
    else:
        st.success("✨ The queue is currently empty!")


# ============================================================
# GET TOKEN
# ============================================================
elif page == "🎫 Get Token":
    st.markdown("## 🎫 Get Your Digital Token")

    html(
        """
        <div class="glass-card">
            🎀 Join the queue digitally and avoid standing in a physical line.
            <br><br>
            You can receive a token and monitor your estimated waiting time from the dashboard.
        </div>
        """
    )

    service = st.selectbox(
        "🏢 Select Service",
        [
            "🏦 Banking",
            "🏥 Hospital",
            "🎓 College Office",
            "🏛️ Government Service",
            "🍔 Canteen",
            "🛒 Customer Service",
        ],
    )

    if st.button("✨ GENERATE DIGITAL TOKEN"):
        token, wait = create_token(service)
        st.success(f"Token {token} generated successfully!")
        html(
            f"""
            <div class="serving-card">
                <div class="serving-title">YOUR DIGITAL TOKEN</div>
                <div class="token-number">{token}</div>
                <div class="live-badge">ESTIMATED WAIT: {wait:.0f} MIN</div>
            </div>
            """
        )
        st.balloons()


# ============================================================
# AI QUEUE VISION
# ============================================================
elif page == "📷 AI Queue Vision":
    st.markdown("## 📷 AI Queue Vision")

    html(
        """
        <div class="glass-card">
            🤖 Use a webcam or upload a queue image.
            <br><br>
            The system can analyse the image and estimate the number of people in the queue.
        </div>
        """
    )

    source = st.radio("Choose Image Source", ["📸 Webcam", "🖼️ Upload Image"], horizontal=True)

    image = None
    if source == "📸 Webcam":
        camera = st.camera_input("Take a picture of the queue")
        if camera:
            image = Image.open(camera)
    else:
        uploaded = st.file_uploader("Upload Queue Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)

    if image:
        st.markdown("### 🔍 Queue Analysis")

        # FIX #6: PNGs can have an alpha channel (RGBA) which crashed cvtColor
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lightweight demo estimate. For accurate counting use YOLO person detection.
        count = 0
        output = img.copy()
        height, width = gray.shape

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 1800 and h > w and h > height * 0.20:
                count += 1
                cv2.rectangle(output, (x, y), (x + w, y + h), (139, 92, 246), 2)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(output, caption="Queue Vision Analysis", **STRETCH)

        with col2:
            html(
                f"""
                <div class="serving-card">
                    <div class="serving-title">PEOPLE DETECTED</div>
                    <div class="token-number">{count}</div>
                    <div class="live-badge">SCAN COMPLETE</div>
                </div>
                """
            )
            predicted = predict_wait_time(count, get_open_counters())
            st.metric("Predicted Wait", f"{predicted:.0f} min")


# ============================================================
# STAFF CONSOLE
# ============================================================
elif page == "🧑‍💼 Staff Console":
    st.markdown("## 🧑‍💼 Staff Control Center")
    st.write("Manage service counters and call customer tokens.")

    # FIX #7: messages are stored, then the page reruns, so the cards
    # always show the up-to-date token (before, they showed the old one).
    if st.session_state.flash:
        kind, message = st.session_state.flash
        getattr(st, kind)(message)
        st.session_state.flash = None

    cols = st.columns(4)

    for index in range(1, 5):
        is_open = st.session_state.counters[index]

        with cols[index - 1]:
            status = "🟢 OPEN" if is_open else "🔴 CLOSED"
            html(
                f"""
                <div class="glass-card">
                    <h3>🏢 Counter {index}</h3>
                    <p>Status: <b>{status}</b></p>
                    <p>Current Token: <b>{get_counter_current(index)}</b></p>
                </div>
                """
            )

            if is_open:
                if st.button("⏭️ Call Next", key=f"next_{index}"):
                    token, already = call_next(index)
                    if token is None:
                        st.session_state.flash = ("info", "No waiting tokens.")
                    elif already:
                        st.session_state.flash = (
                            "warning",
                            f"Counter {index} is still serving {token}. Complete it first.",
                        )
                    else:
                        st.session_state.flash = ("success", f"Counter {index} now serving {token}")
                    st.rerun()

                if st.button("✅ Complete", key=f"complete_{index}"):
                    token = complete_token(index)
                    st.session_state.flash = (
                        ("success", f"{token} completed.")
                        if token
                        else ("info", f"Counter {index} has no active token.")
                    )
                    st.rerun()

                if st.button("🔴 Close Counter", key=f"close_{index}"):
                    if get_counter_current(index) != "A00":
                        st.session_state.flash = (
                            "warning",
                            f"Complete the active token on counter {index} before closing it.",
                        )
                    else:
                        st.session_state.counters[index] = False
                    st.rerun()
            else:
                if st.button("🟢 Open Counter", key=f"open_{index}"):
                    st.session_state.counters[index] = True
                    st.rerun()


# ============================================================
# ANALYTICS
# ============================================================
elif page == "📊 Analytics":
    st.markdown("## 📊 Queue Analytics")

    db = SessionLocal()
    try:
        tokens = db.query(Token).order_by(Token.id.asc()).all()
    finally:
        db.close()

    if not tokens:
        st.info("Generate some tokens first to see analytics.")
    else:
        df = pd.DataFrame(
            [
                {
                    "Token": t.token_number,
                    "Service": t.service,
                    "Status": t.status,
                    "Wait": t.wait_minutes,
                    "Created": t.created_at,
                }
                for t in tokens
            ]
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🎫 Total Tokens", len(df))
        with c2:
            st.metric("✅ Served", int((df["Status"] == "served").sum()))
        with c3:
            st.metric("⏱️ Average Wait", f"{df['Wait'].mean():.1f} min")

        st.markdown("### 📈 Service Usage")
        st.bar_chart(df.groupby("Service").size())

        st.markdown("### 📋 Token Data")
        st.dataframe(df, hide_index=True, **STRETCH)

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "queueless_analytics.csv",
            "text/csv",
        )


# ============================================================
# SYSTEM
# ============================================================
elif page == "⚙️ System":
    st.markdown("## ⚙️ System Information")

    html(
        """
        <div class="glass-card">
            <h3>🤖 Smart QueueLess Technology Stack</h3>
            <p>🐍 <b>Python</b> — Core programming</p>
            <p>🎨 <b>Streamlit</b> — Web interface</p>
            <p>📷 <b>OpenCV</b> — Computer vision</p>
            <p>🤖 <b>YOLO</b> — Person detection (planned upgrade)</p>
            <p>🧠 <b>Scikit-learn</b> — ML wait-time prediction</p>
            <p>📊 <b>Pandas / NumPy</b> — Data processing</p>
            <p>💾 <b>SQLAlchemy / SQLite</b> — Database</p>
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================
html(
    """
    <div style="text-align:center; color:#94a3b8; padding:35px 10px 15px 10px; font-size:13px;">
        🎫 <b style="color:#c4b5fd;">SMART QUEUELESS COUNTER</b>
        <br>
        AI • Computer Vision • Smart Tokens • Predictive Analytics
        <br><br>
        Built with Python + Streamlit ✨
    </div>
    """
)