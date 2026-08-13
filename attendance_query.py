import sqlite3
from pathlib import Path


class AttendanceQuery:


   
    def __init__(self, db_path="./db/attendance.db"):
        self.db_path = Path(db_path)

    def _connect(self):
        if not self.db_path.exists():
            return None
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    
    def studentHistory(self, student_id):
    
        connection = self._connect()
        if connection is None:
            return []
        try:
            rows = connection.execute(
                "SELECT date, status FROM attendance WHERE student_id = ?",
                (student_id,),
            ).fetchall()
        finally:
            connection.close()

        def sortKey(row):
            parts = row[0].split(".")
            return (parts[2], parts[1], parts[0]) if len(parts) == 3 else (row[0],)

        return sorted(rows, key=sortKey)

    def batchRates(self, batch_name=None):
        
        connection = self._connect()
        if connection is None:
            return []
        try:
            query = """
                SELECT s.student_id, s.name,
                       SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) AS attended,
                       COUNT(*) AS total
                FROM attendance a
                JOIN student s ON s.student_id = a.student_id
                JOIN batch b ON b.id = s.batch_id
            """
            params = ()
            if batch_name:
                query += " WHERE b.name = ?"
                params = (batch_name,)
            query += " GROUP BY s.student_id ORDER BY s.name"
            return connection.execute(query, params).fetchall()
        finally:
            connection.close()
