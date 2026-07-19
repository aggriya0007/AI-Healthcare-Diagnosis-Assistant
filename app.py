from pydoc import doc

from flask import Flask, render_template, request, redirect, session, send_file, abort
import pandas as pd
import qrcode
import joblib
import sqlite3
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)

app = Flask(__name__)
app.secret_key = "healthcare_ai_project"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "healthcare.db")
LOGO_PATH = os.path.join(BASE_DIR, "static", "images", "logo.png")
QR_PATH = os.path.join(BASE_DIR, "static", "images", "qr.png")
REPORT_PATH = os.path.join(BASE_DIR, "medical_report.pdf")

# Your live Render URL (used to build the QR verification link)
BASE_URL = "https://ai-healthcare-diagnosis-assistant.onrender.com"

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
# Database Setup (self-healing: safe to run every startup)
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

    # Safe migration: add report_id column if it doesn't already exist
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
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


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

    report_id = "AIH-" + datetime.now().strftime("%Y%m%d%H%M%S")

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

    now = datetime.now()

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
# History Page  (this is now a proper top-level route)
# ==========================
@app.route("/history")
def history():
    conn = get_db_connection()
    cursor = conn.cursor()

    name = request.args.get("name", "")
    disease = request.args.get("disease", "")
    gender = request.args.get("gender", "")

    query = """
        SELECT *
        FROM patients
        WHERE 1=1
    """
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

    # ==========================
    # Disease Dropdown
    # ==========================
    cursor.execute("""
        SELECT DISTINCT disease
        FROM patients
        ORDER BY disease
    """)
    diseases = [row["disease"] for row in cursor.fetchall()]

    # ==========================
    # Statistics
    # ==========================
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
# Dashboard
# ==========================
@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()

    # KPI Cards
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

    # Doughnut Chart — Disease Distribution
    cursor.execute("SELECT disease, COUNT(*) FROM patients GROUP BY disease")
    rows = cursor.fetchall()
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    # Bar Chart — Average Confidence by Disease
    cursor.execute("SELECT disease, AVG(confidence) FROM patients GROUP BY disease")
    confidence_rows = cursor.fetchall()
    confidence_labels = [r[0] for r in confidence_rows]
    confidence_values = [round(r[1], 2) for r in confidence_rows]

    # Line Chart — Daily Patients
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
# Download Professional PDF Report
# ==========================
from reportlab.pdfgen import canvas
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(550, 20, f"Page {page_num}")


@app.route("/download_report")
def download_report():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Prefer the patient from the current session (the one who just got a
    # prediction); fall back to the most recent record in the database.
    report_id = session.get("report_id")
    latest_patient = None

    if report_id:
        cursor.execute(
            "SELECT * FROM patients WHERE report_id = ? ORDER BY id DESC LIMIT 1",
            (report_id,),
        )
        latest_patient = cursor.fetchone()

    if latest_patient is None:
        cursor.execute("SELECT * FROM patients ORDER BY id DESC LIMIT 1")
        latest_patient = cursor.fetchone()

    if latest_patient is None:
        conn.close()
        return "No patient records found yet. Please complete a prediction first.", 404

    # Use the report_id already stored on the patient record, so the
    # verification QR code and the file name both point to the SAME id.
    report_id = latest_patient["report_id"] or (
        "AIH-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )

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
        SELECT name, age, gender, disease, confidence, date, time
        FROM patients ORDER BY id DESC
    """)
    records = cursor.fetchall()

    # ==========================
    # Statistics
    # ==========================
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE gender='Male'")
    male_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patients WHERE gender='Female'")
    female_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT disease) FROM patients")
    total_diseases = cursor.fetchone()[0]

    conn.close()

    # -------------------------
    # Generate QR code first (must exist on disk before we build the PDF)
    # -------------------------
    os.makedirs(os.path.dirname(QR_PATH), exist_ok=True)
    qr_data = f"{BASE_URL}/verify/{report_id}"
    qr_img = qrcode.make(qr_data)
    qr_img.save(QR_PATH)

    # -------------------------
    # Build PDF
    # -------------------------
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
        logo = Image(LOGO_PATH, width=2.2 * inch, height=2.2 * inch)
        logo.hAlign = "CENTER"
        elements.append(logo)
        elements.append(Spacer(1, 10))

    # Header
    elements.append(
        Paragraph(
            f"<b>Report ID:</b> {report_id}",
            styles["Normal"]
        )
    )
    elements.append(Paragraph(
        "<font color='#0d6efd'><b><font size='24'>AI Healthcare Diagnosis Assistant</font></b></font>",
        title,
    ))
    elements.append(Paragraph(
        "<font size='15'><b>Machine Learning Based Disease Prediction System</b></font>",
        heading,
    ))
    elements.append(Paragraph(
        "<font color='grey'><i>Professional Patient Medical Report</i></font>",
        heading,
    ))
    elements.append(Spacer(1, 10))

    header_table = Table([
        ["Report ID", report_id],
        ["Generated On", datetime.now().strftime("%d-%m-%Y")],
        ["Generated Time", datetime.now().strftime("%I:%M %p")],
    ], colWidths=[150, 250])

    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 20))

    # Report Summary
    elements.append(Paragraph(
        "<b><font size='16' color='#198754'>Report Summary</font></b>", heading
    ))

    summary_data = [
        ["Total Patients", total_patients],
        ["Average Confidence", f"{avg_conf}%"],
        ["Most Common Disease", common_disease],
    ]
    summary_table = Table(summary_data, colWidths=[220, 220])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#0d6efd")),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#198754")),
        ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#6f42c1")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Patient Summary
    elements.append(Paragraph(
        "<b><font size='15' color='#0d6efd'>Patient Summary</font></b>", styles["Heading2"]
    ))
    elements.append(Spacer(1, 10))

    patient_summary_table = Table([
        ["Patient Name", latest_patient["name"]],
        ["Age", latest_patient["age"]],
        ["Gender", latest_patient["gender"]],
        ["Phone", latest_patient["phone"]],
        ["Email", latest_patient["email"]],
        ["Report Date", datetime.now().strftime("%d-%m-%Y")],
        ["Generated By", "AI Healthcare Diagnosis Assistant"],
    ], colWidths=[170, 280])

    patient_summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_summary_table)
    elements.append(Spacer(1, 20))

    # Latest Diagnosis
    elements.append(Paragraph(
        "<font size='16' color='#dc3545'><b>Latest Diagnosis</b></font>", heading
    ))
    elements.append(Spacer(1, 10))

    disease = latest_patient["disease"]
    doctor = doctor_dict.get(disease, "General Physician")
    confidence = latest_patient["confidence"]

    if confidence >= 90:
        risk = "Very High Confidence"
        risk_color = colors.green
    elif confidence >= 75:
        risk = "High Confidence"
        risk_color = colors.orange
    else:
        risk = "Low Confidence"
        risk_color = colors.red

    latest_table = Table([
        ["Patient Name", latest_patient["name"]],
        ["Age", latest_patient["age"]],
        ["Gender", latest_patient["gender"]],
        ["Predicted Disease", disease],
        ["Confidence", f"{latest_patient['confidence']}%"],
        ["Recommended Doctor", doctor],
        ["Risk Level", risk],
    ], colWidths=[180, 250])

    latest_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dc3545")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(latest_table)
    elements.append(Spacer(1, 20))

    # Full Patient Table
    data = [["Name", "Age", "Gender", "Disease", "Confidence", "Date"]]
    for r in records:
        data.append([r["name"], r["age"], r["gender"], r["disease"],
                     f"{r['confidence']}%", r["date"]])

    patient_table = Table(data, colWidths=[100, 40, 55, 120, 70, 80])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#198754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.beige]),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 25))

    # ==========================
    # QR Code Verification
    # ==========================
    elements.append(
        Paragraph(
            "<font size='15' color='#0d6efd'><b>Report Verification</b></font>",
            heading
        )
    )

    elements.append(
        Paragraph(
            "Scan the QR code below to verify the authenticity of this medical report.",
            normal
        )
    )

    elements.append(Spacer(1, 10))

    qr_image = Image(
        QR_PATH,
        width=1.5 * inch,
        height=1.5 * inch
    )

    qr_image.hAlign = "CENTER"

    elements.append(qr_image)

    elements.append(Spacer(1, 20))

    # ==========================
    # Confidential Notice
    # ==========================
    elements.append(
        Paragraph(
            "<font color='#dc3545'><b>CONFIDENTIAL MEDICAL REPORT</b></font>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "This report contains confidential patient information. It is intended only for the patient and authorized healthcare professionals. Unauthorized sharing, copying, or distribution is prohibited.",
            normal
        )
    )

    elements.append(Spacer(1, 15))

    # ==========================
    # Disclaimer
    # ==========================
    elements.append(
        Paragraph(
            "<b>Disclaimer:</b> This report has been generated using an Artificial Intelligence based disease prediction system. The prediction is intended for educational and preliminary assessment purposes only and should not be considered a substitute for diagnosis or treatment by a qualified medical professional.",
            normal
        )
    )

    elements.append(Spacer(1, 20))

    # Footer
    elements.append(Spacer(1, 25))

    elements.append(
        Paragraph(
            """
            <b>AI Healthcare Diagnosis Assistant</b><br/>
            Machine Learning Based Disease Prediction System<br/><br/>

            Developed by Aggriya Anand<br/><br/>

            <font color='grey'>
            This report is automatically generated using Artificial Intelligence.
            It should not replace professional medical advice.
            </font>
            """,
            styles["Italic"]
        )
    )

    # -------------------------
    # Build PDF
    # -------------------------
    doc.build(elements)

    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name=f"{report_id}.pdf"
    )


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