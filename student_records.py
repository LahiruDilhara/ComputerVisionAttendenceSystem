"""
student_records.py

Reads the same info.xml that sams.py reads, and recovers the session date from
a sheet filename using the same ``DD.MM.YYYY.bNN`` convention.

Kept in its own module so that Parts II, III and IV all agree on who the
students are and what order they appear in.  The row order in this file *is*
the row order on the printed sheet, which is what lets an extracted signature
be attributed to a student.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FILENAME_PATTERN = re.compile(r"((\d\d\.){2}\d{4})\.(b\d+)")


class StudentRecords:
    """Students of one batch, in printed sheet order."""

    def __init__(self, xml_path="public/xml/info.xml", batch_name=None):
        self.xml_path = Path(xml_path)
        self.batches = self.parseXml()
        self.batch_name = batch_name or self.defaultBatch()
        if self.batch_name not in self.batches:
            print(f"batch '{self.batch_name}' not found in {self.xml_path}")
            sys.exit(1)
        self.students = self.batches[self.batch_name]

    def parseXml(self):
        tree = ET.parse(str(self.xml_path))
        root = tree.getroot()
        batches = {}
        batches_node = root.find("batches")
        if batches_node is None:
            return batches
        for batch_node in batches_node:
            students = []
            students_node = batch_node.find("students")
            if students_node is not None:
                for student_node in students_node.findall("student"):
                    index_node = student_node.find("index")
                    name_node = student_node.find("name")
                    if index_node is not None and name_node is not None:
                        students.append({
                            "index": (index_node.text or "").strip(),
                            "name": (name_node.text or "").strip(),
                        })
            batches[batch_node.tag] = students
        return batches

    def defaultBatch(self):
        return next(iter(self.batches), "")

    # ------------------------------------------------------------------ #
    def resolveStudent(self, query):
        """Find a student by full index, or by a unique partial match.

        The coursework invokes the tool as ``python investigate.py 001`` while
        real indices are eight digits, so a partial query is accepted as long
        as exactly one student matches it.  An ambiguous query lists the
        candidates rather than guessing.
        """
        query = str(query).strip()
        exact = [s for s in self.students if s["index"] == query]
        if exact:
            return exact[0], self.students.index(exact[0])

        partial = [s for s in self.students if query and query in s["index"]]
        if len(partial) == 1:
            return partial[0], self.students.index(partial[0])
        return None, partial

    @staticmethod
    def dateFromFilename(path):
        """Return the ``DD.MM.YYYY`` date encoded in a sheet filename."""
        match = FILENAME_PATTERN.fullmatch(Path(path).stem)
        return match.group(1) if match else None

    @staticmethod
    def batchFromFilename(path):
        match = FILENAME_PATTERN.fullmatch(Path(path).stem)
        return match.group(3) if match else None
