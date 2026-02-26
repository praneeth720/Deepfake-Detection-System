import sqlite3
from datetime import datetime

DB_NAME = "deepfake_system.db"

# =========================
# DATABASE CONNECTION
# =========================

def get_connection():
    return sqlite3.connect(DB_NAME)

# =========================
# INITIALIZE DATABASE
# =========================

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            file_type TEXT NOT NULL,
            confidence REAL,
            verdict TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    # Default admin (plain text password)
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password, role)
        VALUES (?, ?, ?)
    """, ("admin", "1234", "admin"))

    conn.commit()
    conn.close()

# =========================
# AUTHENTICATION
# =========================

def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, password)
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]

    return None

# =========================
# USER REGISTRATION
# =========================

def register_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, password, "user")
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# =========================
# LOGGING SYSTEM
# =========================

def save_log(username, file_type, confidence, verdict):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (username, file_type, confidence, verdict, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        username,
        file_type,
        confidence,
        verdict,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

# =========================
# ADMIN DASHBOARD
# =========================

def get_all_logs():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT username, file_type, confidence, verdict, timestamp
        FROM logs
        ORDER BY timestamp DESC
    """)

    logs = cursor.fetchall()
    conn.close()

    return [
        {
            "username": log[0],
            "file_type": log[1],
            "confidence": log[2],
            "verdict": log[3],
            "timestamp": log[4]
        }
        for log in logs
    ]