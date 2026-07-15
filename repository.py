import sqlite3
import os
from dataclasses import dataclass

class Database():
    def __init__(self, db_path="./db/attendance.db"):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.initTables()
    
    def initTables(self):
        print("Initializing tables in database")
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS student (
            student_id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            batch_id INTEGER NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batch(id) ON DELETE CASCADE ON UPDATE CASCADE
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT CHECK(status IN ('Present', 'Absent')),
            PRIMARY KEY(student_id, date),
            FOREIGN KEY (student_id) REFERENCES student(student_id) ON DELETE CASCADE ON UPDATE CASCADE
        )
        """)
        self.connection.commit()

@dataclass
class Student:
    student_id: str
    name: str
    batch_id: int

@dataclass
class Batch:
    id: int
    name: str

@dataclass
class Attendance:
    student_id: str
    date: str
    status: str

class AttendanceRepository:
    def __init__(self, db_path="./db/attendance.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = Database(db_path=db_path)

    def get_or_create_batch(self, batch_name: str) -> int:
        self.db.cursor.execute("SELECT id FROM batch WHERE name = ?", (batch_name,))
        row = self.db.cursor.fetchone()
        if row:
            return row[0]
        else:
            self.db.cursor.execute("INSERT INTO batch (name) VALUES (?)", (batch_name,))
            self.db.connection.commit()
            return self.db.cursor.lastrowid

    def get_or_create_student(self, student_id: str, name: str, batch_id: int) -> Student:
        self.db.cursor.execute("SELECT student_id, name, batch_id FROM student WHERE student_id = ?", (student_id,))
        row = self.db.cursor.fetchone()
        if row:
            return Student(*row)
        else:
            self.db.cursor.execute("INSERT INTO student (student_id, name, batch_id) VALUES (?, ?, ?)", (student_id, name, batch_id))
            self.db.connection.commit()
            return Student(student_id, name, batch_id)

    def markAttendance(self, student_id: str, name: str, batch_name: str, present: bool, date: str):
        status = 'Present' if present else 'Absent'
        
        batch_id = self.get_or_create_batch(batch_name)
        
        student = self.get_or_create_student(student_id, name, batch_id)
        
        self.db.cursor.execute("SELECT status FROM attendance WHERE student_id = ? AND date = ?", (student_id, date))
        row = self.db.cursor.fetchone()
        
        if row:
            self.db.cursor.execute("UPDATE attendance SET status = ? WHERE student_id = ? AND date = ?", (status, student_id, date))
        else:
            self.db.cursor.execute("INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)", (student_id, date, status))
            
        self.db.connection.commit()
