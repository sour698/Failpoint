import streamlit as st
import pickle
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from explain_failure import explain_failure
from auth import signup, login
from history import save_history, get_user_history, clear_user_history

# ---------------- PAGE CONFIG (ONLY ONCE) ----------------
st.set_page_config(
    page_title="AI Interview Failure Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "show_auth" not in st.session_state:
    st.session_state.show_auth = False

# Initialize input states
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "self_reflection" not in st.session_state:
    st.session_state.self_reflection = ""

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# ---------------- TOP BAR ----------------
top = st.columns([7, 2])

with top[1]:
    if not st.session_state.logged_in:
        if st.button("🔐 Sign In / Sign Up"):
            st.session_state.show_auth = True
    else:
        st.success("Logged In")

# ---------------- AUTH MODAL ----------------
if st.session_state.show_auth and not st.session_state.logged_in:
    st.markdown("## 🔐 Account Access")

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if not email or not password:
                st.error("Please enter both email and password")
            elif login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_auth = False
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if not new_email or not new_password or not confirm:
                    st.error("Please fill in all fields")
            elif new_password != confirm:
                    st.error("Passwords do not match")
            else:
                signup_success, msg = signup(new_email, new_password)
                if signup_success:
                      st.success("Account created. Please login.")
                else:
                     st.error(msg)

    st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Info")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as:\n{st.session_state.user_email}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

    st.sidebar.subheader("📜 History")
    history = get_user_history(st.session_state.user_email)

    if history:
        for h in history[-5:][::-1]:
            st.sidebar.write(f"• {h['timestamp']} — {h['Predicted_Failure']}")

        if st.sidebar.button("🗑️ Clear History"):
            clear_user_history(st.session_state.user_email)
            st.sidebar.success("History cleared")
            st.rerun()
    else:
        st.sidebar.caption("No history yet")
else:
    st.sidebar.info("Sign in to unlock analysis & history")

st.sidebar.info("""
• Resume–JD analysis  
• Failure reason prediction  
• Skill gap detection  
• Explainable AI  
""")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model/failure_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- MAIN TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🤖 AI Interview Failure Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Understand why a candidate failed and how to improve</p>", unsafe_allow_html=True)
st.divider()

# ---------------- 🔄 REFRESH BUTTON (ADDED) ----------------
if st.button("🔄 New Entry / Refresh Inputs"):
    # Clear text area widget states
    st.session_state.resume_text = ""
    st.session_state.jd_text = ""
    st.session_state.self_reflection = ""

    st.session_state.resume_text_input = ""
    st.session_state.jd_text_input = ""
    st.session_state.self_reflection_input = ""

    # Reset file uploader
    st.session_state.file_uploader_key += 1

    st.rerun()
# ---------------- SAMPLE INPUT ----------------
if st.button("📄 Load Example Input"):
    st.session_state.resume_text = "I know Python, pandas and basic ML."
    st.session_state.jd_text = "Looking for ML engineer with Python, TensorFlow, SQL."
    st.session_state.self_reflection = "I struggled to explain deep learning."
    
    # Also set the text area widget values
    st.session_state.resume_text_input = "I know Python, pandas and basic ML."
    st.session_state.jd_text_input = "Looking for ML engineer with Python, TensorFlow, SQL."
    st.session_state.self_reflection_input = "I struggled to explain deep learning."
    st.rerun()

# ---------------- INPUT SECTION ----------------
disabled = not st.session_state.logged_in

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Resume")

    # Use dynamic key for file uploader
    pdf_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"],
        disabled=disabled,
        key=f"uploaded_pdf_{st.session_state.file_uploader_key}"
    )

    resume_text = ""
    if pdf_file:
        reader = PdfReader(pdf_file)
        resume_text = " ".join([page.extract_text() or "" for page in reader.pages])
        # If user uploads a new PDF, update the resume text
        if resume_text:
            st.session_state.resume_text = resume_text

    resume_text = st.text_area(
        "Paste Resume Text",
        value=st.session_state.resume_text,
        height=220,
        disabled=disabled,
        key="resume_text_input"
    )

with col2:
    st.subheader("📌 Job Description")
    jd_text = st.text_area(
        "Paste Job Description",
        value=st.session_state.jd_text,
        height=220,
        disabled=disabled,
        key="jd_text_input"
    )

st.subheader("🧠 Self Reflection (Optional)")
self_reflection = st.text_area(
    "Interview experience",
    value=st.session_state.self_reflection,
    height=120,
    disabled=disabled,
    key="self_reflection_input"
)

# ---------------- ANALYZE ----------------
st.divider()

if st.button("🔍 Analyze Interview", use_container_width=True):

    if not st.session_state.logged_in:
        st.warning("Please Sign In to analyze")
        st.session_state.show_auth = True
        st.stop()

    if not resume_text.strip() or not jd_text.strip():
        st.error("Resume and JD cannot be empty")
        st.stop()

    with st.spinner("Analyzing..."):
        result = explain_failure(
            resume_text,
            jd_text,
            self_reflection,
            model,
            vectorizer
        )

    save_history(
        st.session_state.user_email,
        {
            "Predicted_Failure": result["Predicted_Failure"],
            "Confidence": result["Confidence"],
            "Resume_JD_Match": result["Resume_JD_Match"]
        }
    )

    st.success("Analysis Complete")

    colA, colB, colC = st.columns(3)
    colA.metric("Failure Reason", result["Predicted_Failure"])
    colB.metric("Confidence", f"{result['Confidence']}%")
    colC.metric("Resume–JD Match", f"{result['Resume_JD_Match']}%")

    # ---------- CONFIDENCE GAUGE ----------
    st.subheader("🎯 Confidence Gauge")
    confidence = result["Confidence"]

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.pie(
        [confidence, 100 - confidence],
        startangle=90,
        colors=["#4CAF50", "#E0E0E0"],
        wedgeprops={"width": 0.25}
    )
    ax.text(0, 0, f"{confidence}%", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)

    # ---------- SKILL GAPS ----------
    st.subheader("🔥 Missing Skills")
    skills = result["Missing_Skills"]
   
    if skills:
       # Calculate percentage gap (example: using confidence or fixed percentages)
       # You can modify this logic based on your actual gap calculation
       gap_percentages = []
       for i, skill in enumerate(skills):
           # Example: Distribute the missing percentage (100 - Resume_JD_Match) among skills
           base_gap = (100 - result["Resume_JD_Match"]) / len(skills)
           # Add some variation
           gap_pct = round(base_gap * (0.8 + 0.4 * (i/len(skills))), 1)
           gap_percentages.append(gap_pct)
       
       # Display skills with percentages
       for skill, gap_pct in zip(skills, gap_percentages):
           col1, col2 = st.columns([3, 1])
           with col1:
               st.write(f"**{skill}**")
           with col2:
               st.metric(label="Gap", value=f"{gap_pct}%", label_visibility="collapsed")
    else:
        st.success("No skill gaps detected")


    # ---------- EXPLANATION ----------
    st.subheader("📝 Explanation")
    st.info(result["Explanation_Text"])

    # ---------- PDF ----------
    def generate_pdf(res):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(temp.name, pagesize=A4)
        text = c.beginText(40, 800)
        text.setFont("Helvetica", 11)

        for line in [
            "AI Interview Failure Report",
            "",
            f"Failure Reason: {res['Predicted_Failure']}",
            f"Confidence: {res['Confidence']}%",
            f"Resume-JD Match: {res['Resume_JD_Match']}%",
            "",
            "Missing Skills:",
            ", ".join(res["Missing_Skills"]) or "None",
            "",
            "Explanation:",
            res["Explanation_Text"]
        ]:
            text.textLine(line)

        c.drawText(text)
        c.save()
        return temp.name

    pdf_path = generate_pdf(result)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇ Download PDF Report",
            f,
            file_name="Interview_Analysis_Report.pdf",
            mime="application/pdf"
        )

# ---------------- FOOTER ----------------
st.divider()
st.markdown("<p style='text-align:center;'>Built with ❤️ using Machine Learning & Streamlit</p>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>© 2025 Sourav Das. All rights reserved.</p>",
    unsafe_allow_html=True
)