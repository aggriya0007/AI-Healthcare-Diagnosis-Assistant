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

# All timestamps in this app are shown in Indian Standard Time (IST),
# regardless of what timezone the server (e.g. Render) itself runs in.
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
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)

app = Flask(__name__)


# ==========================
# TEMPORARY DEBUG ERROR HANDLER
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

# Your live Render URL (used to build the QR verification link)
BASE_URL = "https://ai-healthcare-diagnosis-assistant.onrender.com"

# ==========================
# Admin Login Credentials
# ==========================
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# ==========================
# Email (SMTP) Configuration
# ==========================
MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.environ.get("MAIL_PORT", "465"))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")

# ==========================
# Load Machine Learning Model
# ==========================
model = joblib.load("model/disease_model.pkl")

# ==========================
# Load Training Dataset
# ==========================
df = pd.read_csv("dataset/Training.csv")

if "Unnamed: 133" in df.columns:
    df = df.drop(columns=["Unnamed: 133"])

feature_names = df.drop("prognosis", axis=1).columns.tolist()

# ==========================
# Load Disease Descriptions
# ==========================
description_df = pd.read_csv("dataset/symptom_Description.csv")
description_dict = {
    row["Disease"]: row["Description"] for _, row in description_df.iterrows()
}

# ==========================
# Load Disease Precautions
# ==========================
precaution_df = pd.read_csv("dataset/symptom_precaution.csv")
precaution_dict = {
    row["Disease"]: [
        row["Precaution_1"], row["Precaution_2"],
        row["Precaution_3"], row["Precaution_4"]
    ]
    for _, row in precaution_df.iterrows()
}

# ==========================
# Doctor Recommendation
# ==========================
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
# Database Setup (self-healing)
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
# Admin Login Decorator
# ==========================
def login_required(view_func):
    """Protects admin-only pages (Dashboard, History, Patient Details, Batch Download)."""
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("is_admin"):
            session["next_url"] = request.path
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped_view


# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Admin Login / Logout
# ==========================
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


# ==========================
# Patient Registration
# ==========================
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


# ==========================
# Prediction Page
# ==========================
@app.route("/predict")
def predict():
    return render_template("predict.html", symptoms=feature_names)


# ==========================
# Result Page
# ==========================
@app.route("/result", methods=["POST"])
def result():
    user_input = [1 if symptom in request.form else 0 for symptom in feature_names]

    prediction = model.predict([user_input])[0]
    probabilities = model.predict_proba([user_input])[0]
    confidence = round(max(probabilities) * 100, 2)

    description = description_dict.get(prediction, "No description available.")

    precautions = precaution_dict.get(prediction, [
        "Consult a healthcare professional.",
        "Take adequate rest.",
        "Stay hydrated.",
        "Follow medical advice.",
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
        session.get("name"),
        session.get("age"),
        session.get("gender"),
        session.get("phone"),
        session.get("email"),
        ", ".join(symptoms_selected),
        prediction,
        confidence,
        now.strftime("%d-%m-%Y"),
        now.strftime("%H:%M:%S"),
        report_id,
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


# ==========================
# History Page (admin only)
# ==========================
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
        "history.html",
        records=records,
        name=name,
        disease=disease,
        gender=gender,
        diseases=diseases,
        total_patients=total_patients,
        male_patients=male_patients,
        female_patients=female_patients,
        total_diseases=total_diseases
    )


# ==========================
# Patient Details Page (admin only)
# ==========================
@app.route("/patient/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    conn = get_db_connection()
    patient = conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    conn.close()

    if patient is None:
        abort(404)

    doctor = doctor_dict.get(patient["disease"], "General Physician")
    qr_data = f"{BASE_URL}/verify/{patient['report_id']}"

    return render_template(
        "patient_detail.html",
        patient=patient,
        doctor=doctor,
        qr_data=qr_data,
        mail_configured=bool(MAIL_USERNAME and MAIL_PASSWORD),
    )


# ==========================
# Dashboard (admin only)
# ==========================
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

    cursor.execute("""
        SELECT disease, COUNT(*) FROM patients
        GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 1
    """)
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
        "dashboard.html",
        total_patients=total_patients,
        avg_conf=avg_conf,
        common_disease=common_disease,
        labels=labels,
        values=values,
        confidence_labels=confidence_labels,
        confidence_values=confidence_values,
        date_labels=date_labels,
        patient_counts=patient_counts,
    )


# ==========================
# Individual Patient PDF Builder
# ==========================
def build_individual_report_pdf(patient):
    """
    Builds a strictly private medical report PDF containing ONLY 
    the specific patient's information and diagnosis.
    """
    report_id = patient["report_id"] or ("AIH-" + now_ist().strftime("%Y%m%d%H%M%S"))

    # Generate QR code for verification
    os.makedirs(os.path.dirname(QR_PATH), exist_ok=True)
    qr_data = f"{BASE_URL}/verify/{report_id}"
    qr_img = qrcode.make(qr_data)
    qr_img.save(QR_PATH)

    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    title = styles["Title"]
    title.alignment = TA_CENTER
    heading = styles["Heading2"]
    heading.alignment = TA_CENTER
    normal = styles["BodyText"]

    elements = []

    # Logo
    if os.path.exists(LOGO_PATH):
        logo = Image(LOGO_PATH, width=1.8 * inch, height=1.8 * inch)
        logo.hAlign = "CENTER"
        elements.append(logo)
        elements.append(Spacer(1, 10))

    # Header
    elements.append(Paragraph(
        "<font color='#0d6efd'><b><font size='22'>AI Healthcare Diagnosis Assistant</font></b></font>",
        title,
    ))
    elements.append(Paragraph(
        "<font size='13'><b>Individual Patient Medical Report</b></font>",
        heading,
    ))
    elements.append(Spacer(1, 15))

    # Patient Details Table
    elements.append(Paragraph("<b><font size='14' color='#0d6efd'>1. Patient Profile</font></b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    patient_table = Table([
        ["Report ID", report_id],
        ["Patient Name", patient["name"]],
        ["Age / Gender", f"{patient['age']} / {patient['gender']}"],
        ["Phone Number", patient["phone"]],
        ["Email Address", patient["email"]],
        ["Report Date & Time", f"{patient['date']} at {patient['time']}"],
    ], colWidths=[150, 300])

    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 15))

    # Diagnosis Details Table
    elements.append(Paragraph("<b><font size='14' color='#dc3545'>2. Diagnosis Details</font></b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    disease = patient["disease"]
    doctor = doctor_dict.get(disease, "General Physician")
    confidence = patient["confidence"]

    if confidence >= 90:
        risk = "Very High Confidence"
    elif confidence >= 75:
        risk = "High Confidence"
    else:
        risk = "Low Confidence"

    symptoms_text = patient["symptoms"] if patient["symptoms"] else "None selected"

    diagnosis_table = Table([
        ["Reported Symptoms", symptoms_text],
        ["Predicted Disease", disease],
        ["Confidence Level", f"{confidence}% ({risk})"],
        ["Recommended Specialist", doctor],
    ], colWidths=[150, 300])

    diagnosis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dc3545")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 20))

    # Verification QR Code
    elements.append(Paragraph("<b><font size='14' color='#198754'>3. Report Verification</font></b>", styles["Heading2"]))
    elements.append(Paragraph("Scan the QR code below using a smartphone to verify the authenticity of this report.", normal))
    elements.append(Spacer(1, 8))

    qr_image = Image(QR_PATH, width=1.4 * inch, height=1.4 * inch)
    qr_image.hAlign = "CENTER"
    elements.append(qr_image)
    elements.append(Spacer(1, 15))

    # Notices
    elements.append(Paragraph("<font color='#dc3545'><b>CONFIDENTIAL MEDICAL RECORD</b></font>", styles["Heading2"]))
    elements.append(Paragraph("This document contains private medical information belonging exclusively to the patient. Unauthorized access, sharing, or distribution is prohibited.", normal))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Disclaimer:</b> This report is generated using an AI preliminary assessment system and is not an official medical diagnosis. Please consult a qualified doctor.", normal))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>AI Healthcare Diagnosis Assistant</b><br/>Developer: Aggriya Anand", styles["Italic"]))

    doc.build(elements)
    return report_id


# ==========================
# Admin Master PDF Builder (All Patients)
# ==========================
def build_admin_master_pdf():
    """
    Builds a single master report containing overall statistics
    and all patient records for the admin.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT ROUND(AVG(confidence), 2) FROM patients")
    avg_conf = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT disease, COUNT(*) FROM patients
        GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 1
    """)
    row = cursor.fetchone()
    common_disease = row[0] if row else "N/A"

    cursor.execute("""
        SELECT report_id, name, age, gender, disease, confidence, date
        FROM patients ORDER BY id DESC
    """)
    records = cursor.fetchall()
    conn.close()

    doc = SimpleDocTemplate(
        ADMIN_REPORT_PATH,
        pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    title = styles["Title"]
    title.alignment = TA_CENTER
    heading = styles["Heading2"]
    heading.alignment = TA_CENTER

    elements = []

    # Title
    elements.append(Paragraph(
        "<font color='#0d6efd'><b><font size='22'>AI Healthcare - Admin Master Patient Report</font></b></font>",
        title,
    ))
    elements.append(Paragraph(
        f"<font size='11' color='grey'>Generated On: {now_ist().strftime('%d-%m-%Y %I:%M %p')}</font>",
        heading,
    ))
    elements.append(Spacer(1, 15))

    # Summary Statistics Table
    elements.append(Paragraph("<b><font size='14' color='#198754'>System Statistics</font></b>", heading))
    elements.append(Spacer(1, 6))

    summary_table = Table([
        ["Total Registered Patients", total_patients],
        ["Average Diagnosis Confidence", f"{avg_conf}%"],
        ["Most Frequent Condition", common_disease],
    ], colWidths=[220, 220])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Full Patient List Table
    elements.append(Paragraph("<b><font size='14' color='#0d6efd'>All Registered Patients</font></b>", heading))
    elements.append(Spacer(1, 8))

    data = [["Report ID", "Name", "Age/Gender", "Disease", "Conf.", "Date"]]
    for r in records:
        data.append([
            r["report_id"] or "N/A",
            r["name"],
            f"{r['age']}/{r['gender']}",
            r["disease"],
            f"{r['confidence']}%",
            r["date"]
        ])

    patient_table = Table(data, colWidths=[110, 100, 70, 120, 50, 70])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#198754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.beige]),
    ]))
    elements.append(patient_table)

    doc.build(elements)


# ==========================
# Download Individual Patient PDF Report
# ==========================
@app.route("/download_report")
def download_report():
    report_id = session.get("report_id")

    conn = get_db_connection()
    cursor = conn.cursor()

    if report_id:
        cursor.execute(
            "SELECT * FROM patients WHERE report_id = ? ORDER BY id DESC LIMIT 1",
            (report_id,),
        )
        patient = cursor.fetchone()
    else:
        cursor.execute("SELECT * FROM patients ORDER BY id DESC LIMIT 1")
        patient = cursor.fetchone()

    conn.close()

    if patient is None:
        return "No patient record found. Please complete a prediction first.", 404

    report_id = build_individual_report_pdf(patient)

    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name=f"Medical_Report_{report_id}.pdf"
    )


# ==========================
# Download All Patient Reports (Admin Only)
# ==========================
@app.route("/admin/download_all_reports")
@login_required
def admin_download_all_reports():
    build_admin_master_pdf()
    
    filename = f"Master_Patient_Report_{now_ist().strftime('%Y%m%d')}.pdf"
    return send_file(
        ADMIN_REPORT_PATH,
        as_attachment=True,
        download_name=filename
    )


# ==========================
# Email PDF Report to Patient (Admin Only)
# ==========================
@app.route("/email_report/<int:patient_id>", methods=["POST"])
@login_required
def email_report(patient_id):
    conn = get_db_connection()
    patient = conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    conn.close()

    if patient is None:
        abort(404)

    if not patient["email"]:
        return "This patient has no email address on file.", 400

    if not MAIL_USERNAME or not MAIL_PASSWORD:
        return (
            "Email sending isn't configured yet. Set MAIL_USERNAME and MAIL_PASSWORD.",
            500,
        )

    try:
        report_id = build_individual_report_pdf(patient)

        msg = EmailMessage()
        msg["Subject"] = f"Your Medical Report - {report_id}"
        msg["From"] = MAIL_USERNAME
        msg["To"] = patient["email"]
        msg.set_content(
            f"Dear {patient['name']},\n\n"
            f"Please find attached your medical report from the "
            f"AI Healthcare Diagnosis Assistant.\n\n"
            f"Report ID: {report_id}\n"
            f"Predicted Condition: {patient['disease']}\n\n"
            f"This report was generated by an AI prediction system for preliminary "
            f"assessment only. Please consult a qualified doctor.\n\n"
            f"Regards,\nAI Healthcare Diagnosis Assistant"
        )

        with open(REPORT_PATH, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="pdf",
                filename=f"{report_id}.pdf",
            )

        with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as smtp:
            smtp.login(MAIL_USERNAME, MAIL_PASSWORD)
            smtp.send_message(msg)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<h2>Could not send email</h2>"
            f"<p><b>Error:</b> {e}</p>"
            f"<p><a href='/patient/{patient_id}'>&larr; Back to patient</a></p>",
            500,
        )

    return redirect(url_for("patient_detail", patient_id=patient_id, emailed=1))


# ==========================
# QR Verification Route
# ==========================
@app.route("/verify/<report_id>")
def verify(report_id):
    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM patients WHERE report_id = ?", (report_id,)
    ).fetchone()
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
        <p>This report is generated by AI Healthcare Diagnosis Assistant.</p>
        <p>Developed by <b>Aggriya Anand</b></p>
    </body>
    </html>
    """


# ==========================
# Run Application
# ==========================
print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)