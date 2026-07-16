import sqlite3

conn = sqlite3.connect("database/healthcare.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
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

conn.commit()
conn.close()

print("Database Ready!")