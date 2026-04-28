import streamlit as st
import pandas as pd
import joblib
import json, os
from datetime import datetime
from genai_helper import explain_prediction
import re



# ================= GEMINI SETUP =================
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="SafeGuard AI", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "English"
def translate_text(text, lang):

    lang_map = {
        "English": "English",
        "Hindi": "Hindi",
        "Marathi": "Marathi",
        "Kannada": "Kannada"
    }

    if lang == "English":
        return text

    try:
        res = llm.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate this text to {lang_map[lang]}:\n{text}"
        )

        return res.text if res and res.text else text

    except:
        return text

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from io import BytesIO

def create_pdf_report(final_pred, prob, explanation):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # ================= LOGO =================
    try:
        logo = Image("logo.png", width=100, height=60)
        story.append(logo)
    except:
        pass

    # ================= TITLE =================
    title_style = ParagraphStyle(
        name="Title",
        fontSize=18,
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=10
    )
    story.append(Paragraph("🛡️ SafeGuard AI - Safety Risk Report", title_style))
    story.append(Spacer(1, 10))

    # ================= SUMMARY TABLE =================
    table_data = [
        ["Field", "Value"],
        ["Prediction", final_pred],
        ["Confidence Score", f"{prob:.2f}"]
    ]

    table = Table(table_data, colWidths=[200, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # ================= CHART =================
    drawing = Drawing(400, 200)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.height = 125
    chart.width = 300

    chart.data = [[prob * 100]]
    chart.categoryAxis.categoryNames = ["Risk Score"]

    drawing.add(chart)
    story.append(Paragraph("<b>📊 Risk Visualization</b>", styles["Heading3"]))
    story.append(drawing)
    story.append(Spacer(1, 20))

    # ================= MULTI PAGE START =================
    story.append(PageBreak())

    # ================= SECTION STYLE =================
    box_style = ParagraphStyle(
        name="Box",
        backColor=colors.whitesmoke,
        borderWidth=1,
        borderColor=colors.grey,
        borderPadding=10,
        spaceAfter=10
    )

    # ================= CLEAN TEXT =================
    explanation = explanation.replace("###", "")
    explanation = explanation.replace("**", "")

    sections = {
        "🔍 Root Cause": "Root Cause",
        "⚠ Risk Factors": "Risk Factors",
        "🔁 Chances of Recurrence": "Chances of Recurrence",
        "🛠 Prevention & Solutions": "Prevention & Solutions",
        "🏥 Worker Treatment": "Worker Treatment",
        "📊 Long-term Safety Measures": "Long-term Safety Measures"
    }

    # ================= SECTION LOOP =================
    for icon_title, sec in sections.items():
        if sec in explanation:
            parts = explanation.split(sec, 1)
            explanation = parts[1]

            # Find next section
            next_pos = len(explanation)
            for s in sections.values():
                if s in explanation:
                    pos = explanation.find(s)
                    if pos != -1:
                        next_pos = min(next_pos, pos)

            content = explanation[:next_pos]

            # Section title
            story.append(Paragraph(f"<b>{icon_title}</b>", styles["Heading2"]))
            story.append(Spacer(1, 6))

            # Box content
            for line in content.split("."):
                if line.strip():
                    story.append(Paragraph(line.strip() + ".", box_style))

            story.append(Spacer(1, 10))

    # ================= FOOTER =================
    footer = ParagraphStyle(
        name="Footer",
        fontSize=9,
        textColor=colors.grey
    )

    story.append(Spacer(1, 20))
    story.append(Paragraph("Generated by SafeGuard AI", footer))

    doc.build(story)
    buffer.seek(0)

    return buffer
# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    data = joblib.load("models/final_model.pkl")
    return data["pipeline"], data["model"], data["columns"]

pipeline, model, columns = load_model()

# ==============================
# USER SYSTEM
# ==============================
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        json.dump({}, open(USER_FILE, "w"))
    return json.load(open(USER_FILE))

def save_users(users):
    json.dump(users, open(USER_FILE, "w"))

# ==============================
# SESSION
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ==============================
# STYLE (LOGIN BACKGROUND)
# ==============================
def login_style():
    st.markdown("""
    <style>
    .stApp {
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.9)),
    url("https://images.unsplash.com/photo-1503387762-592deb58ef4e");
    background-size: cover;
}

    .card {
        background: rgba(0,0,0,0.85);
        padding:40px;
        border-radius:20px;
        box-shadow:0px 0px 40px rgba(0,0,0,0.6);
        text-align:center;
    }

    .stButton>button {
        width:100%;
        background:linear-gradient(135deg,#2563eb,#1d4ed8);
        color:white;
        border-radius:10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOGIN PAGE (CENTERED)
# ==============================
def login():
    login_style()

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("## 🛡️ SafeGuard AI")
        st.write("Industrial Safety Platform")

        email = st.text_input("📧 Email")
        pwd = st.text_input("🔑 Password", type="password")

        if st.button("Login 🚀"):
            users = load_users()

            if email not in users:
                st.error("Email not registered")
                return

            if isinstance(users[email], str):
                if users[email] != pwd:
                    st.error("Wrong password")
                    return
            else:
                if users[email]["password"] != pwd:
                    st.error("Wrong password")
                    return

            st.session_state.logged_in = True
            st.session_state.page = "home"
            st.rerun()

        if st.button("✨ Create Account"):
            st.session_state.page = "signup"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# SIGNUP
# ==============================
def signup():
    login_style()

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("## Create Account")

        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")

        if st.button("Register"):
            users = load_users()
            users[email] = pwd
            save_users(users)

            st.success("Account created")
            st.session_state.page = "login"
            st.rerun()

        if st.button("Back"):
            st.session_state.page = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# NAVBAR
# ==============================
def navbar():
    col1, col2, col3, col4 = st.columns([4,1,1,1])

    with col1:
        st.markdown("### 🛡️ SafeGuard AI")

    with col2:
        if st.button("Home"):
            st.session_state.page = "home"

    with col3:
        if st.button("Predict"):
            st.session_state.page = "predict"

    with col4:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

# ==============================
# HOME PAGE
# ==============================
def home():
    navbar()

    # ================= PREMIUM STYLE =================
    st.markdown("""
    <style>

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #0f172a, #020617);
        padding:70px;
        border-radius:20px;
        position:relative;
        overflow:hidden;
    }

    .hero::before {
        content:"";
        position:absolute;
        width:400px;
        height:400px;
        background:radial-gradient(circle, #3b82f6, transparent);
        top:-100px;
        right:-100px;
        opacity:0.3;
    }

    .title {
        font-size:48px;
        font-weight:700;
        letter-spacing:-1px;
    }

    .highlight {
        color:#3b82f6;
    }

    /* GLASS CARD */
    .glass {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-radius:15px;
        padding:25px;
        border:1px solid rgba(255,255,255,0.1);
        transition:0.3s;
    }

    .glass:hover {
        transform:translateY(-5px);
        box-shadow:0 10px 30px rgba(59,130,246,0.3);
    }

    /* BUTTON */
    .stButton>button {
        background: linear-gradient(135deg,#2563eb,#1d4ed8);
        color:white;
        border-radius:12px;
        padding:10px 20px;
        transition:0.3s;
    }

    .stButton>button:hover {
        transform:scale(1.05);
        box-shadow:0 0 20px rgba(59,130,246,0.6);
    }

    /* BADGE */
    .badge {
        background:rgba(255,0,0,0.15);
        color:#f87171;
        padding:6px 12px;
        border-radius:20px;
        font-size:12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ================= HERO =================
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("""
        <div class="hero">
            <div class="title">
            Predict workplace risk <br>
            <span class="highlight">before it happens.</span>
            </div>
            <br>
            <p>AI-powered industrial safety platform that helps prevent accidents before they occur.</p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)

        with colA:
            if st.button("🚀 Start Prediction"):
                st.session_state.page = "predict"

        with colB:
            if st.button("💬 Ask AI"):
                st.session_state.page = "chat"

    # ================= SIDE PANEL =================
    with col2:
        st.markdown("""
        <div class="glass">
            <h4>📊 Live Risk Overview</h4>
            <p>✔ 4 Risk Categories</p>
            <p>✔ Multi Industry Coverage</p>
            <p>✔ AI Insights Enabled</p>
            <hr>
            <span class="badge">⚠ High Risk Alert</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= INDUSTRIES =================
    st.markdown("### 🏭 Industries We Serve")

    industries = ["Construction","Manufacturing","Healthcare","Logistics","Energy","Government"]
    cols = st.columns(len(industries))

    for i, ind in enumerate(industries):
        cols[i].markdown(f"""
        <div class="glass">
            🏢<br><b>{ind}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= FEATURES =================
    st.markdown("### 🚀 Platform Features")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("""
    <div class="glass">
    📊<br><b>Risk Prediction</b><br>
    Predict incidents before they occur
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("""
    <div class="glass">
    📄<br><b>AI Reports</b><br>
    Auto-generated safety insights
    </div>
    """, unsafe_allow_html=True)

    c3.markdown("""
    <div class="glass">
    🌍<br><b>Multilingual</b><br>
    Works across global teams
    </div>
    """, unsafe_allow_html=True)

    c4.markdown("""
    <div class="glass">
    🤖<br><b>AI Chatbot</b><br>
    Ask safety-related queries
    </div>
    """, unsafe_allow_html=True)
# PREDICTOR
# ==============================
def predictor():
    navbar()
    
    if "explanation" not in st.session_state:
        st.session_state.explanation = ""

    lang = st.selectbox(
        "🌐 Select Language",
        ["English", "Hindi", "Marathi", "Kannada"]
        )
    st.markdown("""
    <style>
    .section {
        background: rgba(255,255,255,0.05);
        padding:20px;
        border-radius:12px;
        margin-bottom:15px;
    }

    .result-box {
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:20px;
        font-weight:bold;
    }

    .high-risk {
        background: rgba(255,0,0,0.15);
        color:#f87171;
    }

    .low-risk {
        background: rgba(34,197,94,0.15);
        color:#4ade80;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📊 Risk Predictor")

    # ================= FORM =================
    with st.form("prediction_form"):

        # ================= COMPANY =================
        st.markdown("### 🏢 Company Information")

        col10, col11 = st.columns(2)

        with col10:
            company = st.text_input("Company Name *")
            industry = st.text_input("Industry *")

        with col11:
            size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
            size_map = {"Small": 1, "Medium": 2, "Large": 3}

            employees = st.number_input("Employees", min_value=1)
            hours = st.number_input("Total Hours Worked", min_value=1)

        # ================= JOB =================
        st.markdown("### 👷 Job Information")

        col1, col2 = st.columns(2)

        with col1:
            job = st.text_input("Job Description *")

        with col2:
            soc_desc = st.text_input("SOC Description *")
            soc_code = st.text_input("SOC Code *")   # ✅ ADDED

        # ================= INCIDENT =================
        st.markdown("### ⚠ Incident Details")

        desc = st.text_area("What happened *")
        before = st.text_area("Before incident *")

        col3, col4 = st.columns(2)

        with col3:
            location = st.text_input("Location *")
            injury = st.text_input("Injury *")

        with col4:
            obj = st.text_input("Object/Substance *")

        # ================= TIME =================
        st.markdown("### ⏰ Incident Timing")

        col5, col6, col7 = st.columns(3)

        with col5:
            incident_hour = st.slider("Incident Hour", 0, 23)

        with col6:
            start_hour = st.slider("Shift Start Hour", 0, 23)

        with col7:
            time_unknown = st.selectbox("Time Known?", ["Known Time", "Unknown"])

        # ================= WORK =================
        st.markdown("### 📉 Work Impact")

        col8, col9 = st.columns(2)

        with col8:
            dafw = st.number_input("Days Away From Work", 0)

        with col9:
            djtr = st.number_input("Job Transfer Days", 0)

        # ================= EXTRA =================
        st.markdown("### 📌 Additional Details")

        col12, col13 = st.columns(2)

        with col12:
            establishment = st.selectbox("Establishment Type", ["Private Industry", "Government", "Local government"])
            soc_reviewed = st.selectbox("SOC Reviewed", ["Reviewed", "Not Reviewed", "Not SOC coded"])

        with col13:
            naics_code = st.text_input("NAICS Code *")
            naics_year = st.text_input("NAICS Year *")

        soc_prob = st.slider("SOC Probability", 0.0, 1.0, 0.5)

        incident_type = st.selectbox("Incident Type", [
            "Injury", "Skin disorder", "Respiratory condition",
            "Poisoning", "Hearing Loss", "All other illness"
        ])

        submit = st.form_submit_button("🚀 Predict Risk")

        # ================= SUBMIT =================
        if submit:

            # ✅ VALIDATION
            required_fields = {
                "Job Description": job,
                "SOC Description": soc_desc,
                "SOC Code": soc_code,
                "What happened": desc,
                "Before incident": before,
                "Location": location,
                "Injury": injury,
                "Object/Substance": obj,
                "Company": company,
                "Industry": industry,
                "NAICS Code": naics_code,
                "NAICS Year": naics_year
            }

            missing = [name for name, value in required_fields.items() if not value or str(value).strip() == ""]

            if missing:
                st.warning(f"⚠ Please fill required fields: {', '.join(missing)}")
                return

            # ================= DATA =================
            df = pd.DataFrame([{
                "date_of_incident": datetime.today(),
                "incident_hour": incident_hour,
                "start_hour": start_hour,
                "dafw_num_away": dafw,
                "djtr_num_tr": djtr,
                "job_description": job,
                "soc_description": soc_desc,
                "soc_code": soc_code,
                "NEW_INCIDENT_DESCRIPTION": desc,
                "NEW_NAR_WHAT_HAPPENED": desc,
                "NEW_NAR_BEFORE_INCIDENT": before,
                "NEW_INCIDENT_LOCATION": location,
                "NEW_NAR_INJURY_ILLNESS": injury,
                "NEW_NAR_OBJECT_SUBSTANCE": obj,
                "company_name": company,
                "industry_description": industry,
                "size": size_map[size],
                "annual_average_employees": employees,
                "total_hours_worked": hours,
                "soc_probability": soc_prob,
                "naics_code": naics_code,
                "naics_year": naics_year,
                "type_of_incident": incident_type,
                "soc_reviewed": soc_reviewed,
                "time_unknown": time_unknown,
                "establishment_type": establishment
            }])

            # ================= MODEL =================
            processed = pipeline.transform(df)
            X = processed.drop(columns=['incident_outcome'], errors='ignore')

            for col in columns:
                if col not in X:
                    X[col] = 0

            X = X[columns]

            pred = model.predict(X)[0]
            prob = model.predict_proba(X).max()

            # ================= LABEL =================
            incident_outcome_map = {
                1: "Death",
                0: "Days Away From Work",
                2: "Job Transfer / Restriction",
                3: "Other Recordable Case"
            }

            final_pred = incident_outcome_map.get(int(pred), "Unknown")

            # 🔥 SAVE STATE (FIX)
            st.session_state.final_pred = final_pred
            st.session_state.prob = prob
            st.session_state.input_df = df.to_dict(orient="records")[0]

        # ================= SHOW AFTER RERUN =================
    if "final_pred" in st.session_state:

        final_pred = st.session_state.final_pred
        prob = st.session_state.prob

        lang = st.session_state.lang
        translated_pred = translate_text(final_pred, lang)

        st.markdown(f"<h2>{translated_pred}</h2>", unsafe_allow_html=True)

            # ================= COLOR =================
        if final_pred == "Death":
            color = "#ef4444"
        elif final_pred == "Days Away From Work":
            color = "#f59e0b"
        else:
            color = "#22c55e"

            # ================= BOX 1 =================
        st.markdown(f"""
        <div style="
        background: linear-gradient(135deg, #1e293b, #020617);
        padding:25px;
        border-radius:15px;
        border-left:6px solid {color};
        margin-bottom:20px;
        ">
        <h3 style="color:#e2e8f0;">🎯 Prediction Result</h3>

        <h2 style="color:{color}; margin-top:10px;">
        {final_pred}
        </h2>

        <p style="color:#cbd5f5; font-size:18px;">
        📊 Confidence Score: <b>{prob:.2f}</b>
        </p>
        </div>
        """, unsafe_allow_html=True)

            # ================= BUTTON =================
        if st.button("🤖 Generate AI Explanation"):
            with st.spinner("🤖 AI is analyzing the incident..."):
                st.session_state.explanation = explain_prediction(
                    st.session_state.input_df,
                    final_pred,
                    prob
                )

        # ================= SHOW AI =================
        if "explanation" in st.session_state and st.session_state.explanation:

            explanation = st.session_state.explanation
            

            lang = st.session_state.lang
            translated_exp = translate_text(explanation, lang)

            # st.markdown(f"<p>{translated_exp}</p>", unsafe_allow_html=True)

            import re
            explanation = explanation.replace("###", "")
            explanation = explanation.replace("\n", "<br>")
            explanation = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", explanation)

            explanation = explanation.replace("1. Root Cause", "<br><b>🔍 Root Cause</b><br>")
            explanation = explanation.replace("2. Risk Factors", "<br><b>⚠ Risk Factors</b><br>")
            explanation = explanation.replace("3. Chances of Recurrence", "<br><b>🔁 Chances of Recurrence</b><br>")
            explanation = explanation.replace("4. Prevention & Solutions", "<br><b>🛠 Prevention & Solutions</b><br>")
            explanation = explanation.replace("5. Worker Treatment", "<br><b>🏥 Worker Treatment</b><br>")
            explanation = explanation.replace("6. Long-term Safety Measures", "<br><b>📊 Long-term Safety Measures</b><br>")

            st.markdown(f"""
            <div style="
            background: linear-gradient(135deg, #020617, #0f172a);
            padding:25px;
            border-radius:15px;
            border:1px solid rgba(255,255,255,0.1);
            margin-top:20px;
            ">
            <h3 style="color:#e2e8f0;">🤖 AI Safety Explanation</h3>

            <p style="color:#e2e8f0; font-size:16px; line-height:1.7;">
            {explanation}
            </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ================= DOWNLOAD BUTTON =================
        report_text = f"""
        🛡️ SafeGuard AI - Risk Report

        ----------------------------------------
        📊 Prediction Result:
        {st.session_state.final_pred}

        📈 Confidence Score:
        {st.session_state.prob:.2f}

        ----------------------------------------
        🤖 AI Safety Explanation:
        {st.session_state.explanation}

        ----------------------------------------
        Generated by SafeGuard AI
        """

        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name="safety_report.txt",
            mime="text/plain"
        )
        
        # ================= PDF DOWNLOAD =================
        explanation = st.session_state.get("explanation", "")
        pdf_file = create_pdf_report(final_pred, prob, explanation)

        st.download_button(
            label="📄 Download PDF",
            data=pdf_file,
            file_name="safety_report.pdf",
            mime="application/pdf"
        )
# ═══════════════════════════════════════════════════════════
# ASK AI  (Fixed ChatGPT-like behavior)
# ═══════════════════════════════════════════════════════════     
def show_ask_ai():
    navbar()
    st.markdown('<div class="sg-app-page">', unsafe_allow_html=True)

    # ================= INIT =================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_q" not in st.session_state:
        st.session_state.chat_q = ""

    # ✅ NEW FIX
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
        
    if "explanation" not in st.session_state:
        st.session_state.explanation = ""

    st.markdown("""
    <div class="sg-page-hdr">
      <h2>🤖 AI Safety Assistant</h2>
      <p>Ask anything about workplace safety, OSHA regulations, hazard prevention, or incident response</p>
    </div>""", unsafe_allow_html=True)

    # ================= SUGGESTIONS =================
    sugg = [
        "PPE requirements for construction?",
        "How to file OSHA 300 report?",
        "Top causes of workplace fatalities?",
        "Preventing chemical exposure?"
    ]

    sc = st.columns(4)
    for i, s in enumerate(sugg):
        with sc[i]:
            if st.button(s, key=f"sugg_{i}"):
                st.session_state.chat_q = s
                st.rerun()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ================= CHAT HISTORY =================
    if st.session_state.messages:
        st.markdown('<div class="sg-card" style="min-height:200px;">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'''
                <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
                    <div class="sg-bubble-u">{msg["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div style="display:flex;justify-content:flex-start;margin-bottom:8px;">
                    <div class="sg-bubble-a">{msg["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= INPUT FIX =================
    if st.session_state.clear_input:
        st.session_state.chat_q = ""
        st.session_state.clear_input = False

    ic, bc = st.columns([5, 1])

    with ic:
        user_input = st.text_input(
            "",
            placeholder="Ask about safety, OSHA, hazard prevention...",
            key="chat_q",
            label_visibility="collapsed"
        )

    with bc:
        send = st.button("Send →", key="send_ai")

    # ================= PROCESS INPUT =================
    if send and st.session_state.chat_q.strip():

        pending = st.session_state.chat_q.strip()

        # ✅ Save user message
        st.session_state.messages.append({"role": "user", "content": pending})

        import time

        # ===== SINGLE AI CALL (FIXED) =====
        reply = ""
        intent = "GENERAL_CHAT"

        for attempt in range(2):
            try:
                res = llm.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"""
You are a workplace safety AI assistant.

1. First identify intent:
   - RISK_PREDICTION
   - SAFETY_EXPERT_QA
   - GENERAL_CHAT

2. Then give a clear answer.

User Query:
{pending}

Return format:
INTENT: <intent>
ANSWER:
<your answer>
"""
                )

                if res and res.text:
                    output = res.text.strip()

                    # ✅ SPLIT INTENT + ANSWER
                    if "ANSWER:" in output:
                        parts = output.split("ANSWER:")
                        intent = parts[0].replace("INTENT:", "").strip()
                        reply = parts[1].strip()
                    else:
                        reply = output

                    break

            except Exception as e:
                if "429" in str(e):
                    time.sleep(3)
                    reply = "⚠ AI is busy (quota reached). Please try again later."
                else:
                    reply = f"⚠ AI Error: {str(e)}"
                    break

        # ✅ FINAL FALLBACK
        if not reply:
            reply = "⚠ AI is currently unavailable. Please try again later."
        # ✅ Save AI response
        st.session_state.messages.append({"role": "assistant", "content": reply})

        # ✅ SAFE CLEAR (FIX)
        st.session_state.clear_input = True

        st.rerun()

    # ================= CLEAR CHAT =================
    if st.session_state.messages:
        _, cc = st.columns([6, 1])
        with cc:
            if st.button("Clear chat", key="clr_chat"):
                st.session_state.messages = []
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
# ==============================
# ROUTER
# ==============================
if not st.session_state.logged_in:
    if st.session_state.page == "signup":
        signup()
    else:
        login()
else:
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "predict":
        predictor()
    elif st.session_state.page == "chat":
        show_ask_ai()
