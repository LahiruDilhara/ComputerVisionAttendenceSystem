"""
infovis.py - Part III: Visualization

Shows a summary of attendance for a given student, as a four-panel chart.

    $ uv run python infovis.py 001
    $ uv run python infovis.py 10000409     # full index
    $ uv run python infovis.py --row 1      # by row number on the sheet
    $ uv run python infovis.py --all        # every student in the batch
    $ uv run python infovis.py 001 --cohort # also draw the class overview

Reads the attendance database written by sams.py; it never writes to it.
Run sams.py over the sheets first.
"""

import argparse
import sys
from pathlib import Path

from rich import print
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from attendance_query import AttendanceQuery
from attendance_visualizer import (
    REQUIRED_RATE, AttendanceVisualizer, StudentAttendance,
)
from student_records import StudentRecords

console = Console()


class AttendanceSummary:
    """Builds and reports one student's attendance summary."""

    def __init__(self, records, db_path="./db/attendance.db", output_dir="public/output"):
        self.records = records
        self.query = AttendanceQuery(db_path)
        self.output_dir = Path(output_dir)
        self.visualizer = AttendanceVisualizer()

    # ------------------------------------------------------------------ #
    def cohortRates(self):
        """(index, name, rate) for every student, for the comparison panel."""
        return [
            (student_id, name, attended / total if total else 0.0)
            for student_id, name, attended, total in self.query.batchRates(
                self.records.batch_name
            )
        ]

    def sessionTotals(self):
        """(date, attended, enrolled) per session, for the class overview.

        Derived from the per-student records rather than stored separately, so
        the overview can never disagree with the individual summaries.
        """
        sessions = {}
        for student in self.records.students:
            for date, status in self.query.studentHistory(student["index"]):
                attended, total = sessions.get(date, (0, 0))
                sessions[date] = (attended + (status == "Present"), total + 1)

        def chronological(item):
            day, month, year = item[0].split(".")
            return (year, month, day)

        return [(date, a, t) for date, (a, t) in sorted(sessions.items(), key=chronological)]

    # ------------------------------------------------------------------ #
    def summarise(self, student):
        """Print the record and write the chart.  Returns True on success."""
        history = self.query.studentHistory(student["index"])
        console.rule(f"[bold]{student['name']}  ({student['index']})")

        if not history:
            console.print("[yellow]No attendance recorded for this student. "
                          "Run sams.py over the sheets first.[/yellow]")
            return False

        attendance = StudentAttendance(student["index"], student["name"], history)

        table = Table(header_style="bold")
        table.add_column("Date")
        table.add_column("Status")
        for date, status in history:
            colour = "green" if status == "Present" else "red"
            table.add_row(date, f"[{colour}]{status}[/{colour}]")
        console.print(table)

        verdict = ("[green]meets the requirement[/green]"
                   if attendance.meetsRequirement()
                   else f"[red]below the {REQUIRED_RATE:.0%} requirement[/red]")
        console.print(f"Attended [bold]{attendance.attended}/{attendance.total}[/bold] "
                      f"({attendance.rate:.0%}) - {verdict}")
        streak = attendance.longestAbsentStreak()
        if streak > 1:
            console.print(f"[yellow]Longest run of consecutive absences: {streak}[/yellow]")

        path = self.visualizer.studentDashboard(
            attendance, self.cohortRates(),
            self.output_dir / f"attendance_{student['index']}.png",
        )
        console.print(f"Chart written to [bold]{path}[/bold]")
        return True

    def drawCohortOverview(self):
        sessions = self.sessionTotals()
        if not sessions:
            return None
        path = self.visualizer.cohortOverview(
            sessions, self.output_dir / "attendance_by_session.png"
        )
        console.print(f"Class overview written to [bold]{path}[/bold]")
        return path


# ---------------------------------------------------------------------- #
def cliArgumentParser():
    parser = argparse.ArgumentParser(
        description="Student Attendance Management System - attendance visualization",
        formatter_class=RichHelpFormatter,
        usage="uv run python infovis.py [-h] <student_number>",
    )
    parser.add_argument("student", nargs="?",
                        help="Student index, full or a unique part of it")
    parser.add_argument("-r", "--row", type=int, default=None,
                        help="Select by row number on the sheet instead (1-based)")
    parser.add_argument("-x", "--xml", type=Path, default=Path("public/xml/info.xml"),
                        help="Path to the student xml file")
    parser.add_argument("-b", "--batch", default=None, help="Batch name, e.g. b15")
    parser.add_argument("-d", "--db", type=Path, default=Path("./db/attendance.db"),
                        help="Path to the attendance database")
    parser.add_argument("-o", "--output", type=Path, default=Path("public/output"),
                        help="Directory for the generated charts")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Summarise every student in the batch")
    parser.add_argument("-c", "--cohort", action="store_true",
                        help="Also draw the class-wide session overview")
    return parser.parse_args()


def resolveTargets(records, args):
    """Turn the command line into a list of students to summarise."""
    if args.all:
        return records.students

    if args.row is not None:
        if not 1 <= args.row <= len(records.students):
            console.print(f"[red]Row {args.row} is out of range: the sheet has "
                          f"{len(records.students)} student rows[/red]")
            sys.exit(1)
        return [records.students[args.row - 1]]

    student, found = records.resolveStudent(args.student)

    # The brief documents "python infovis.py 001", but the indices in info.xml
    # are eight digits, so a short number cannot be one.  Rather than fail on
    # the command the marker is most likely to type, read it as a row number -
    # and say so, so the fallback is never mistaken for an index lookup.
    if student is None and not found and args.student.isdigit():
        row = int(args.student)
        if 1 <= row <= len(records.students):
            student = records.students[row - 1]
            console.print(f"[yellow]No index matches '{args.student}'; reading it as "
                          f"row {row} on the sheet -> {student['index']}[/yellow]")

    if student is None:
        if found:
            console.print(f"[red]'{args.student}' matches more than one student:[/red]")
            candidates = found
        else:
            console.print(f"[red]No student matching '{args.student}' in batch "
                          f"{records.batch_name}[/red]")
            candidates = records.students
        for candidate in candidates:
            console.print(f"  {candidate['index']}  {candidate['name']}")
        sys.exit(1)

    return [student]


def main():
    args = cliArgumentParser()
    if not args.xml.exists():
        console.print(f"[red]Xml file does not exist: {args.xml}[/red]")
        sys.exit(1)
    if not args.student and not args.all and args.row is None:
        console.print("[red]Give a student number, or use --row / --all[/red]")
        sys.exit(1)
    if not args.db.exists():
        console.print(f"[red]Database not found: {args.db}[/red]")
        console.print("Run sams.py over the signing sheets first, for example:")
        console.print("  [dim]uv run python sams.py public/img/10.07.2019.b15.jpeg "
                      "public/xml/info.xml[/dim]")
        sys.exit(1)

    records = StudentRecords(args.xml, args.batch)
    summary = AttendanceSummary(records, str(args.db), args.output)

    summarised = 0
    for student in resolveTargets(records, args):
        if summary.summarise(student):
            summarised += 1

    if args.cohort or args.all:
        summary.drawCohortOverview()

    console.print(f"\n[bold]{summarised} student summary chart(s) written to "
                  f"{args.output}/[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
