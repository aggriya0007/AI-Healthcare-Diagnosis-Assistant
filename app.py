from flask import Flask, render_template, request, redirect, session
import pandas as pd
import qrcode
from reportlab.lib.units import inch
import joblib
import sqlite3
import os
from datetime import datetime
from flask import send_file
from reportlab.platypus import Image
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)
app.secret_key = "healthcare_ai_project"

# ==========================
# Load Machine Learning Model
# ==========================
model = joblib.load("model/disease_model.pkl")

# ==========================
# Load Training Dataset
# ==========================
df = pd.read_csv("dataset/Training.csv")

# Remove unwanted column if present
if "Unnamed: 133" in df.columns:
    df = df.drop(columns=["Unnamed: 133"])

# Feature Names
feature_names = df.drop("prognosis", axis=1).columns.tolist()

# ==========================
# Load Disease Descriptions
# ==========================
description_df = pd.read_csv("dataset/symptom_Description.csv")

description_dict = {}

for _, row in description_df.iterrows():
    description_dict[row["Disease"]] = row["Description"]

# ==========================
# Load Disease Precautions
# ==========================
precaution_df = pd.read_csv("dataset/symptom_precaution.csv")

precaution_dict = {}

for _, row in precaution_df.iterrows():
    precaution_dict[row["Disease"]] = [
        row["Precaution_1"],
        row["Precaution_2"],
        row["Precaution_3"],
        row["Precaution_4"]
    ]
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
    "Arthritis": "Orthopedic Specialist"
}

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
    return render_template(
        "predict.html",
        symptoms=feature_names
    )

# ==========================
# Result Page
# ==========================
@app.route("/result", methods=["POST"])
def result():

    user_input = []

    for symptom in feature_names:
        if symptom in request.form:
            user_input.append(1)
        else:
            user_input.append(0)

    # Predict disease
    prediction = model.predict([user_input])[0]


     # Confidence
    probabilities = model.predict_proba([user_input])[0]
    confidence = round(max(probabilities) * 100, 2)

     # Description
    description = description_dict.get(
    prediction,
    "No description available."
    )

     # Precautions
    precautions = precaution_dict.get(
    prediction,
    [
        "Consult a healthcare professional.",
        "Take adequate rest.",
        "Stay hydrated.",
        "Follow medical advice."
    ]
    )

     # Doctor Recommendation
    doctor = doctor_dict.get(
    prediction,
    "General Physician"
   )

    # Save in Session
    session["prediction"] = prediction
    session["confidence"] = confidence
    session["description"] = description
    session["doctor"] = doctor
    session["precautions"] = precautions


    # ==========================
    # Save Prediction to Database
    # ==========================

    symptoms_selected = []

    for symptom in feature_names:
        if symptom in request.form:
            symptoms_selected.append(
                symptom.replace("_", " ").title()
            )

    db_path = os.path.abspath("database/healthcare.db")
    print("Using database:", db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    now = datetime.now()

    cursor.execute("""
    INSERT INTO patients
    (name, age, gender, phone, email, symptoms, disease, confidence, date, time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        now.strftime("%H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        disease=prediction,
        confidence=confidence,
        description=description,
        precautions=precautions,
        doctor=doctor
    )
@app.route("/history")
def history():

    conn = sqlite3.connect("database/healthcare.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM patients
    ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )
# ==========================
# Dashboard
# ==========================
@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("database/healthcare.db")
    cursor = conn.cursor()

    # ======================
    # KPI Cards
    # ======================

    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM patients")
    avg_conf = cursor.fetchone()[0]

    if avg_conf is None:
        avg_conf = 0

    avg_conf = round(avg_conf, 2)

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM patients
        GROUP BY disease
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    if row:
        common_disease = row[0]
    else:
        common_disease = "No Data"

    # ======================
    # Doughnut Chart
    # Disease Distribution
    # ======================

    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM patients
        GROUP BY disease
    """)

    rows = cursor.fetchall()

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    # ======================
    # Bar Chart
    # Average Confidence by Disease
    # ======================

    cursor.execute("""
        SELECT disease, AVG(confidence)
        FROM patients
        GROUP BY disease
    """)

    confidence_rows = cursor.fetchall()

    confidence_labels = [r[0] for r in confidence_rows]
    confidence_values = [round(r[1], 2) for r in confidence_rows]

    # ======================
    # Line Chart
    # Daily Patients
    # ======================

    cursor.execute("""
        SELECT date, COUNT(*)
        FROM patients
        GROUP BY date
        ORDER BY date
    """)

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
        patient_counts=patient_counts
    )
# ==========================
# Download Professional PDF Report
# ==========================
@app.route("/download_pdf")
def download_pdf():

    from flask import send_file
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        Image
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    import sqlite3
    import os
    from datetime import datetime
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("BASE_DIR =", BASE_DIR)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(BASE_DIR, "static", "images", "logo.png")

    print("Logo Path:", logo_path)
    print("Exists:", os.path.exists(logo_path))

    # -------------------------
    # Fetch Data
    # -------------------------

    conn = sqlite3.connect("database/healthcare.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
        name,
        age,
        gender,
        disease,
        confidence,
        date,
        time
        FROM patients
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    # -------------------------
    # Latest Patient
    # -------------------------

    cursor.execute("""
    SELECT
    name,
    age,
    gender,
    phone,
    email,
    disease,
    confidence
    FROM patients
    ORDER BY id DESC
    LIMIT 1
    """)

    latest_patient = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT ROUND(AVG(confidence),2) FROM patients")
    avg_conf = cursor.fetchone()[0]

    if avg_conf is None:
        avg_conf = 0

    cursor.execute("""
        SELECT disease,COUNT(*)
        FROM patients
        GROUP BY disease
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    row = cursor.fetchone()

    if row:
        common_disease = row[0]
    else:
        common_disease = "N/A"

    conn.close()

    # -------------------------
    # PDF
    # -------------------------

    pdf_path = "medical_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title = styles["Title"]
    title.alignment = TA_CENTER

    heading = styles["Heading2"]
    heading.alignment = TA_CENTER

    normal = styles["BodyText"]

    elements = []

    # -------------------------
    # Logo
    # -------------------------

    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    logo_path = os.path.join(BASE_DIR, "static", "images", "logo.png")

    print("Logo Path:", logo_path)
    print("Exists:", os.path.exists(logo_path))

    import os

    logo_path = os.path.abspath("static/images/logo.png")

    print("Logo Path:", logo_path)
    print("Exists:", os.path.exists(logo_path))

    if os.path.exists(logo_path):

        logo = Image(
            logo_path,
            width=2.2*inch,
            height=2.2*inch
        )

        logo.hAlign = "CENTER"

        elements.append(logo)
        elements.append(Spacer(1,10))

    # -------------------------
    # Professional Header
    # -------------------------

    report_id = "AIH-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    elements.append(
    Paragraph(
        "<font color='#0d6efd'><b><font size='24'>AI Healthcare Diagnosis Assistant</font></b></font>",
        title
    )
    )

    elements.append(
    Paragraph(
        "<font size='15'><b>Machine Learning Based Disease Prediction System</b></font>",
        heading
    )
    )

    elements.append(
    Paragraph(
        "<font color='grey'><i>Professional Patient Medical Report</i></font>",
        heading
    )
    )

    elements.append(Spacer(1,10))

    header_table = Table([
    ["Report ID", report_id],
    ["Generated On", datetime.now().strftime("%d-%m-%Y")],
    ["Generated Time", datetime.now().strftime("%I:%M %p")]
    ], colWidths=[150, 250])

    header_table.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#0d6efd")),
    ("TEXTCOLOR",(0,0),(0,-1),colors.white),
    ("BACKGROUND",(1,0),(1,-1),colors.whitesmoke),
    ("GRID",(0,0),(-1,-1),0.8,colors.grey),
    ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),
    ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ("TOPPADDING",(0,0),(-1,-1),8),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1,20))

    # -------------------------
    # Report Summary
    # -------------------------

    elements.append(
    Paragraph(
        "<b><font size='16' color='#198754'>Report Summary</font></b>",
        heading
    )
    )

    elements.append(Spacer(1,10))

    summary_data = [

    ["👥 Total Patients", total_patients],

    ["🎯 Average Confidence", f"{avg_conf}%"],

    ["🩺 Most Common Disease", common_disease]

   ]

    summary_table = Table(
    summary_data,
    colWidths=[220,220]
    )

    summary_table.setStyle(TableStyle([

    ("BACKGROUND",(0,0),(0,0),colors.HexColor("#0d6efd")),
    ("BACKGROUND",(0,1),(0,1),colors.HexColor("#198754")),
    ("BACKGROUND",(0,2),(0,2),colors.HexColor("#6f42c1")),

    ("TEXTCOLOR",(0,0),(0,-1),colors.white),

    ("BACKGROUND",(1,0),(1,-1),colors.beige),

    ("GRID",(0,0),(-1,-1),1,colors.grey),

    ("ALIGN",(0,0),(-1,-1),"CENTER"),

    ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

    ("BOTTOMPADDING",(0,0),(-1,-1),10),

    ("TOPPADDING",(0,0),(-1,-1),10)

    ]))

    elements.append(summary_table)

    elements.append(Spacer(1,20))

    # -------------------------
    # PATIENT SUMMARY
    # -------------------------

    elements.append(
    Paragraph(
        "<b><font size='15' color='#0d6efd'>Patient Summary</font></b>",
        styles["Heading2"]
    )
    )

    elements.append(Spacer(1,10))

    summary_table = Table([
        ["Patient Name", latest_patient[0]],
        ["Age", latest_patient[1]],
        ["Gender", latest_patient[2]],
        ["Phone", latest_patient[3]],
        ["Email", latest_patient[4]],
        ["Report Date", datetime.now().strftime("%d-%m-%Y")],
        ["Generated By", "AI Healthcare Diagnosis Assistant"]
        ], colWidths=[170, 280])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0,0), (0,-1), colors.white),
        ("BACKGROUND", (1,0), (1,-1), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ]))

    elements.append(summary_table)
    elements.append(Spacer(1,20))

    # -------------------------
    # Latest Diagnosis
    # -------------------------

    elements.append(
    Paragraph(
        "<font size='16' color='#dc3545'><b>Latest Diagnosis</b></font>",
        heading
    )
   )

    elements.append(Spacer(1,10))

    if latest_patient:

        disease = latest_patient[5]

        doctor = doctor_dict.get(
        disease,
        "General Physician"
    )

        latest_table = Table([
    
            ["Patient Name", latest_patient[0]],

            ["Age", latest_patient[1]],

            ["Gender", latest_patient[2]],

            ["Predicted Disease", disease],

            ["Confidence", f"{latest_patient[6]}%"],

            ["Recommended Doctor", doctor]

    ], colWidths=[180,250])

    latest_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#dc3545")),

        ("TEXTCOLOR",(0,0),(0,-1),colors.white),

        ("BACKGROUND",(1,0),(1,-1),colors.whitesmoke),

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ("TOPPADDING",(0,0),(-1,-1),8)

    ]))

    elements.append(latest_table)

    elements.append(Spacer(1,20))

    # -------------------------
    # Patient Table
    # -------------------------

    data = [[

        "Name",

        "Age",

        "Gender",

        "Disease",

        "Confidence",

        "Date"

    ]]

    for r in records:

        data.append([

            r[0],

            r[1],

            r[2],

            r[3],

            f"{r[4]}%",

            r[5]

        ])

    patient_table = Table(

        data,

        colWidths=[100,40,55,120,70,80]

    )

    patient_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#198754")),

        ("TEXTCOLOR",(0,0),(-1,0),colors.white),

        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("GRID",(0,0),(-1,-1),0.5,colors.grey),

        ("BOTTOMPADDING",(0,0),(-1,0),10),

        ("ROWBACKGROUNDS",(0,1),(-1,-1),

            [colors.whitesmoke, colors.beige]

        )

    ]))

    elements.append(patient_table)

    elements.append(Spacer(1,25))

    # -------------------------
    # Footer
    # -------------------------

    elements.append(

        Paragraph(

            "<font color='grey'><b>Confidential Medical Report</b></font>",

            normal

        )

    )

    elements.append(

        Paragraph(

            "Generated by AI Healthcare Diagnosis Assistant",

            normal

        )

    )

    elements.append(

        Paragraph(

            "<b>Developed by Aggriya Anand</b>",

            normal

        )

    )


    # ---------------------------
    # Step 1: Create Report ID
    # ---------------------------

    report_id = datetime.now().strftime("%Y%m%d%H%M%S")


    # -------------------------
    # Build PDF
    # -------------------------
    # ---------------------------
        # ---------------------------
    # Generate QR Code
    # ---------------------------

    qr_data = """
    AI Healthcare Diagnosis Assistant

    Patient Name: {}
    Disease: {}
    Confidence: {}%

    Generated by Aggriya Anand
    """.format(
        session.get("name", "User"),
        session.get("prediction", "No diagnosis"),
        session.get("confidence", 0)
    )


    qr = qrcode.make(qr_data)


    qr_path = os.path.join(
        BASE_DIR,
        "static",
        "images",
        "qr.png"
    )


    qr.save(qr_path)


    # Add QR into PDF

    qr_image = Image(
        qr_path,
        width=1.2*inch,
        height=1.2*inch
    )


    elements.append(Spacer(1,20))

    elements.append(
        Paragraph(
            "<b>Scan QR Code for Report Verification</b>",
            normal
        )
    )

    elements.append(Spacer(1,10))

    elements.append(qr_image)




    # -------------------------
    # Build PDF
    # -------------------------

    doc.build(elements)

    return send_file(
        pdf_path,
        as_attachment=True
    )
# ==========================
# Run Application
# ==========================

# ==========================
# QR Verification Route
# ==========================

@app.route("/verify/<report_id>")
def verify(report_id):

    return f"""
    <html>
    <body style="font-family: Arial; text-align:center;">

    <h1>✅ Verified Medical Report</h1>

    <h2>AI Healthcare Diagnosis Assistant</h2>

    <p><b>Report ID:</b> {report_id}</p>

    <p>
    This report is generated by AI Healthcare Diagnosis Assistant.
    </p>

    <p>
    Developed by <b>Aggriya Anand</b>
    </p>

    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)