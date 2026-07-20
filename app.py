from pydoc import doc
from flask import Flask, render_template, request, redirect, session, send_file, abort, url_for, flash
import pandas as pd
import qrcode
import joblib
import sqlite3
import os
import smtplib
from functools import wraps
from email.message import EmailMessage
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    """Current date/time, correctly localized to IST."""
    return datetime.now(IST)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
)

app = Flask(__name__)

# ==========================
# DEBUG ERROR HANDLER
# ==========================
import traceback as _traceback

@app.errorhandler(Exception)
def handle_all_errors(e):
    _traceback.print_exc()
    return (
        "<h2>Debug: An error occurred</h2>"
        f"<pre style='white-space: pre-wrap; font-size: 13px;'>{_traceback.format_exc()}</pre>",
        500,
    )

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "healthcare_ai_project")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "healthcare.db")
LOGO_PATH = os.path.join(BASE_DIR, "static", "images", "logo.png")
QR_PATH = os.path.join(BASE_DIR, "static", "images", "qr.png")
REPORT_PATH = os.path.join(BASE_DIR, "medical_report.pdf")
ADMIN_REPORT_PATH = os.path.join(BASE_DIR, "admin_master_report.pdf")

BASE_URL = "https://ai-healthcare-diagnosis-assistant.onrender.com"

# ==========================
# Credentials & Config
# ==========================
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", "465"))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")

# ==========================
# Load Model & Dataset
# ==========================
model = joblib.load("model/disease_model.pkl")

df = pd.read_csv("dataset/Training.csv")
if "Unnamed: 133" in df.columns:
    df = df.drop(columns=["Unnamed: 133"])

feature_names = df.drop("prognosis", axis=1).columns.tolist()

# Descriptions & Precautions
description_df = pd.read_csv("dataset/symptom_Description.csv")
description_dict = {row["Disease"]: row["Description"] for _, row in description_df.iterrows()}

precaution_df = pd.read_csv("dataset/symptom_precaution.csv")
precaution_dict = {
    row["Disease"]: [
        row["Precaution_1"], row["Precaution_2"],
        row["Precaution_3"], row["Precaution_4"]
    ]
    for _, row in precaution_df.iterrows()
}

doctor_dict = {
    "Acne": "Dermatologist",
    "Allergy": "Allergist",
    "Diabetes ": "Endocrinologist",
    "Heart attack": "Cardiologist",
    "Migraine": "Neurologist",
    "Pneumonia": "Pulmonologist",
    "Tuberculosis": "Pulmonologist",
    "Arthritis": "Orthopedic Specialist",
}

# ==========================
# Database Setup
# ==========================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age TEXT,
            gender TEXT,
            phone TEXT,
            email TEXT,
            symptoms TEXT,
            disease TEXT,
            confidence REAL,
            date TEXT,
            time TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(patients)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "report_id" not in existing_cols:
        cursor.execute("ALTER TABLE patients ADD COLUMN report_id TEXT")

    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==========================
# Decorator
# ==========================
def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("is_admin"):
            session["next_url"] = request.path
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped_view

# ==========================
# Helper: Add Patient Elements to Story
# ==========================
def append_patient_report_elements(elements, patient, styles):
    """
    Appends full medical report elements for a single patient into the elements flow.
    """
    title = styles["Title"]
    title.alignment = TA_CENTER
    heading = styles["Heading2"]
    heading.alignment = TA_CENTER
    normal = styles["BodyText"]

    report_id = patient["report_id"] or ("AIH-" + now_ist().strftime("%Y%m%d%H%M%S"))

    # Generate QR Code for patient
    temp_qr_path = os.path.join(BASE_DIR, "static", "images", f"qr_{report_id}.png")
    os.makedirs(os.path.dirname(temp_qr_path), exist_ok=True)
    qr_data = f"{BASE_URL}/verify/{report_id}"
    qr_img = qrcode.make(qr_data)
    qr_img.save(temp_qr_path)

    # Logo
    if os.path.exists(LOGO_PATH):
        logo = Image(LOGO_PATH, width=1.6 * inch, height=1.6 * inch)
        logo.hAlign = "CENTER"
        elements.append(logo)
        elements.append(Spacer(1, 8))

    # Header
    elements.append(Paragraph(
        "<font color='#0d6efd'><b><font size='20'>AI Healthcare Diagnosis Assistant</font></b></font>",
        title,
    ))
    elements.append(Paragraph(
        f"<font size='12'><b>Medical Diagnosis Report ({report_id})</b></font>",
        heading,
    ))
    elements.append(Spacer(1, 12))

    # Patient Profile
    elements.append(Paragraph("<b><font size='13' color='#0d6efd'>1. Patient Profile</font></b>", styles["Heading2"]))
    elements.append(Spacer(1, 4))

    patient_table = Table([
        ["Report ID", report_id],
        ["Patient Name", patient["name"]],
        ["Age / Gender", f"{patient['age']} / {patient['gender']}"],
        ["Phone Number", patient["phone"]],
        ["Email Address", patient["email"]],
        ["Report Date & Time", f"{patient['date']} at {patient['time']}"],
    ], colWidths=[140, 310])

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))

    # Diagnosis Details
    elements.append(Paragraph("<b><font size='13' color='#dc3545'>2. Diagnosis Details</font></b>", styles["Heading2"]))
    elements.append(Spacer(1, 4))

    disease = patient["disease"]
    doctor = doctor_dict.get(disease, "General Physician")
    confidence = patient["confidence"]

    if confidence >= 90:
        risk = "Very High Confidence"
    elif confidence >= 75:
        risk = "High Confidence"
    else:
        risk = "Low Confidence"

    symptoms_text = patient["symptoms"] if patient["symptoms"] else "None recorded"

    diagnosis_table = Table([
        ["Reported Symptoms", symptoms_text],
        ["Predicted Disease", disease],
        ["Confidence Level", f"{confidence}% ({risk})"],
        ["Recommended Specialist", doctor],
    ], colWidths=[140, 310])

    diagnosis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dc3545")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 15))

    # Verification QR Code
    elements.append(Paragraph("<b><font size='13' color='#198754'>3. Report Verification</font></b>", styles["Heading2"]))
    elements.append(Paragraph("Scan QR code below to verify authenticity on live server.", normal))
    elements.append(Spacer(1, 6))

    qr_image = Image(temp_qr_path, width=1.2 * inch, height=1.2 * inch)
    qr_image.hAlign = "CENTER"
    elements.append(qr_image)
    elements.append(Spacer(1, 12))

    # Disclaimers
    elements.append(Paragraph("<font color='#dc3545'><b>CONFIDENTIAL MEDICAL RECORD</b></font>", styles["Heading2"]))
    elements.append(Paragraph("This document contains private medical information belonging exclusively to the patient.", normal))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Disclaimer:</b> Preliminary AI assessment. Please consult a medical professional.", normal))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>AI Healthcare Diagnosis Assistant</b> | Developer: Aggriya Anand", styles["Italic"]))


# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            next_url = session.pop("next_url", None)
            return redirect(next_url or url_for("dashboard"))
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("is_admin", None)
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        session["name"] = request.form["name"]
        session["age"] = request.form["age"]
        session["gender"] = request.form["gender"]
        session["phone"] = request.form["phone"]
        session["email"] = request.form["email"]
        return redirect("/predict")

    return render_template("register.html")

@app.route("/predict")
def predict():
    return render_template("predict.html", symptoms=feature_names)

@app.route("/result", methods=["POST"])
def result():
    user_input = [1 if symptom in request.form else 0 for symptom in feature_names]

    prediction = model.predict([user_input])[0]
    probabilities = model.predict_proba([user_input])[0]
    confidence = round(max(probabilities) * 100, 2)

    description = description_dict.get(prediction, "No description available.")
    precautions = precaution_dict.get(prediction, [
        "Consult a healthcare professional.", "Take adequate rest.", "Stay hydrated.", "Follow medical advice."
    ])
    doctor = doctor_dict.get(prediction, "General Physician")

    report_id = "AIH-" + now_ist().strftime("%Y%m%d%H%M%S")

    session["prediction"] = prediction
    session["confidence"] = confidence
    session["description"] = description
    session["doctor"] = doctor
    session["precautions"] = precautions
    session["report_id"] = report_id

    symptoms_selected = [
        symptom.replace("_", " ").title()
        for symptom in feature_names if symptom in request.form
    ]

    now = now_ist()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients
        (name, age, gender, phone, email, symptoms, disease, confidence, date, time, report_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session.get("name"), session.get("age"), session.get("gender"), session.get("phone"),
        session.get("email"), ", ".join(symptoms_selected), prediction, confidence,
        now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S"), report_id,
    ))
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        disease=prediction,
        confidence=confidence,
        description=description,
        precautions=precautions,
        doctor=doctor,
    )

@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    cursor = conn.cursor()

    name = request.args.get("name", "")
    disease = request.args.get("disease", "")
    gender = request.args.get("gender", "")

    query = "SELECT * FROM patients WHERE 1=1"
    params = []

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if disease:
        query += " AND disease = ?"
        params.append(disease)
    if gender:
        query += " AND gender = ?"
        params.append(gender)

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    records = cursor.fetchall()

    cursor.execute("SELECT DISTINCT disease FROM patients ORDER BY disease")
    diseases = [row["disease"] for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE gender='Male'")
    male_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE gender='Female'")
    female_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT disease) FROM patients")
    total_diseases = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "history.html", records=records, name=name, disease=disease, gender=gender,
        diseases=diseases, total_patients=total_patients, male_patients=male_patients,
        female_patients=female_patients, total_diseases=total_diseases
    )

@app.route("/patient/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    conn = get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()

    if patient is None:
        abort(404)

    doctor = doctor_dict.get(patient["disease"], "General Physician")
    qr_data = f"{BASE_URL}/verify/{patient['report_id']}"

    return render_template(
        "patient_detail.html", patient=patient, doctor=doctor, qr_data=qr_data,
        mail_configured=bool(MAIL_USERNAME and MAIL_PASSWORD),
    )

@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM patients")
    avg_conf = cursor.fetchone()[0] or 0
    avg_conf = round(avg_conf, 2)

    cursor.execute("SELECT disease, COUNT(*) FROM patients GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 1")
    row = cursor.fetchone()
    common_disease = row[0] if row else "No Data"

    cursor.execute("SELECT disease, COUNT(*) FROM patients GROUP BY disease")
    rows = cursor.fetchall()
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    cursor.execute("SELECT disease, AVG(confidence) FROM patients GROUP BY disease")
    confidence_rows = cursor.fetchall()
    confidence_labels = [r[0] for r in confidence_rows]
    confidence_values = [round(r[1], 2) for r in confidence_rows]

    cursor.execute("SELECT date, COUNT(*) FROM patients GROUP BY date ORDER BY date")
    line_rows = cursor.fetchall()
    date_labels = [r[0] for r in line_rows]
    patient_counts = [r[1] for r in line_rows]

    conn.close()

    return render_template(
        "dashboard.html", total_patients=total_patients, avg_conf=avg_conf,
        common_disease=common_disease, labels=labels, values=values,
        confidence_labels=confidence_labels, confidence_values=confidence_values,
        date_labels=date_labels, patient_counts=patient_counts,
    )

# ==========================
# Download Single Patient PDF
# ==========================
@app.route("/download_report")
def download_report():
    report_id = session.get("report_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    if report_id:
        cursor.execute("SELECT * FROM patients WHERE report_id = ? ORDER BY id DESC LIMIT 1", (report_id,))
        patient = cursor.fetchone()
    else:
        cursor.execute("SELECT * FROM patients ORDER BY id DESC LIMIT 1")
        patient = cursor.fetchone()

    conn.close()

    if patient is None:
        return "No patient record found. Please complete a prediction first.", 404

    doc = SimpleDocTemplate(
        REPORT_PATH, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    styles = getSampleStyleSheet()
    elements = []

    append_patient_report_elements(elements, patient, styles)
    doc.build(elements)

    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name=f"Medical_Report_{patient['report_id']}.pdf"
    )

# ==========================
# Download ALL Patients Combined PDF (Admin Only)
# ==========================
@app.route("/admin/download_all_reports")
@login_required
def admin_download_all_reports():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    all_patients = cursor.fetchall()
    conn.close()

    if not all_patients:
        return "No patient records found in database.", 404

    doc = SimpleDocTemplate(
        ADMIN_REPORT_PATH, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
    )
    styles = getSampleStyleSheet()
    elements = []

    # Iterate through all patients and append individual report pages
    for index, patient in enumerate(all_patients):
        append_patient_report_elements(elements, patient, styles)
        
        # Add page break between patients except after the last patient
        if index < len(all_patients) - 1:
            elements.append(PageBreak())

    doc.build(elements)

    filename = f"All_Patient_Medical_Reports_{now_ist().strftime('%Y%m%d')}.pdf"
    return send_file(
        ADMIN_REPORT_PATH,
        as_attachment=True,
        download_name=filename
    )

# ==========================
# Email PDF Route
# ==========================
@app.route("/email_report/<int:patient_id>", methods=["POST"])
@login_required
def email_report(patient_id):
    conn = get_db_connection()
    patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()

    if patient is None:
        abort(404)

    if not patient["email"]:
        return "This patient has no email address on file.", 400

    if not MAIL_USERNAME or not MAIL_PASSWORD:
        return "Email sending isn't configured yet. Set MAIL_USERNAME and MAIL_PASSWORD.", 500

    try:
        doc = SimpleDocTemplate(
            REPORT_PATH, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
        )
        styles = getSampleStyleSheet()
        elements = []
        append_patient_report_elements(elements, patient, styles)
        doc.build(elements)

        report_id = patient["report_id"]

        msg = EmailMessage()
        msg["Subject"] = f"Your Medical Report - {report_id}"
        msg["From"] = MAIL_USERNAME
        msg["To"] = patient["email"]
        msg.set_content(
            f"Dear {patient['name']},\n\n"
            f"Please find attached your medical report.\n\n"
            f"Report ID: {report_id}\n"
            f"Predicted Condition: {patient['disease']}\n\n"
            f"Regards,\nAI Healthcare Diagnosis Assistant"
        )

        with open(REPORT_PATH, "rb") as f:
            msg.add_attachment(
                f.read(), maintype="application", subtype="pdf", filename=f"{report_id}.pdf"
            )

        with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as smtp:
            smtp.login(MAIL_USERNAME, MAIL_PASSWORD)
            smtp.send_message(msg)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h2>Could not send email</h2><p><b>Error:</b> {e}</p>", 500

    return redirect(url_for("patient_detail", patient_id=patient_id, emailed=1))

# ==========================
# QR Verification Route
# ==========================
@app.route("/verify/<report_id>")
def verify(report_id):
    conn = get_db_connection()
    record = conn.execute("SELECT * FROM patients WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()

    if record:
        details = f"""
        <p><b>Patient:</b> {record['name']}</p>
        <p><b>Predicted Disease:</b> {record['disease']}</p>
        <p><b>Confidence:</b> {record['confidence']}%</p>
        <p><b>Date:</b> {record['date']} {record['time']}</p>
        """
    else:
        details = "<p>No matching record found for this report ID.</p>"

    return f"""
    <html>
    <body style="font-family: Arial; text-align:center; padding-top: 40px;">
        <h1>&#9989; Verified Medical Report</h1>
        <h2>AI Healthcare Diagnosis Assistant</h2>
        <p><b>Report ID:</b> {report_id}</p>
        {details}
        <p>Developed by <b>Aggriya Anand</b></p>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)