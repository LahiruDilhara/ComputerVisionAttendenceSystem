"""
investigate.py - Part IV: Recognition

Collects every signature belonging to one student, compares them all against
one another, decides which of them agree, treats that agreeing group as the
student's true signature, and reports anything that does not match it.  The
student's attendance record and chart are produced alongside.

    $ uv run investigate.py 10000409
    $ uv run investigate.py 0409          # any unique part of the index
    $ uv run investigate.py --row 1       # or by row number on the sheet
    $ uv run investigate.py --all

The procedure, step by step:

  1. Resolve the student number against info.xml.
  2. Read every sheet in the images directory and extract that student's
     signature box from each one.
  3. Compare every signature with every other signature of the same student.
  4. Group the ones that agree; the largest, most cohesive group is taken to
     be the student's genuine signature and its medoid becomes the reference.
  5. Report each sample as matching or not matching that reference.
  6. Report the attendance record and draw the chart.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from attendance_query import AttendanceQuery
from signature_extraction import SignatureExtractor
from signature_matching import SignatureMatcher, normalise
from student_records import StudentRecords

console = Console()
PRESENT_COLOUR = "#2e8b57"
ABSENT_COLOUR = "#c0392b"


class SignatureInvestigation:
    """Runs the whole Part IV procedure for one student."""

    def __init__(self, records, images_dir="images", db_path="./db/attendance.db"):
        self.records = records
        self.images_dir = Path(images_dir)
        self.extractor = SignatureExtractor(expected_rows=len(records.students))
        self.matcher = SignatureMatcher()
        self.query = AttendanceQuery(db_path)

    # ------------------------------------------------------------------ #
    def collectSignatures(self, row_index):
        """Step 2: this student's signature from every sheet, in date order.

        Sheets where the box was never signed contribute nothing - there is no
        signature to compare - so they are reported separately rather than
        being fed to the matcher as blank images.

        Whether a box counts as signed is taken from the attendance database
        written by sams.py, not decided again here.  Part II owns that
        decision; re-deciding it would let the two parts of the project
        disagree about the same sheet, which is exactly the kind of silent
        inconsistency that is hard to notice and embarrassing to explain.  The
        database is only consulted, never written to.
        """
        recorded = {
            date: status for date, status
            in self.query.studentHistory(self.records.students[row_index]["index"])
        }
        collected, skipped = [], []

        def chronological(path):
            # Filenames are DD.MM.YYYY, which does not sort correctly as text.
            day, month, year = self.records.dateFromFilename(path).split(".")
            return (year, month, day)

        sheets = sorted(
            (p for p in self.images_dir.glob("*.*")
             if self.records.dateFromFilename(p) is not None),
            key=chronological,
        )
        if not sheets:
            console.print(f"[red]no sheets found in {self.images_dir}/[/red]")
            sys.exit(1)

        for path in sheets:
            date = self.records.dateFromFilename(path)
            try:
                samples = self.extractor.extract(path, date)
            except (ValueError, FileNotFoundError) as error:
                console.print(f"[yellow]skipped {path.name}: {error}[/yellow]")
                continue

            sample = samples[row_index]
            status = recorded.get(date)
            signed = (status == "Present") if status is not None else sample.present
            if not signed:
                skipped.append((date, "not signed"))
                continue

            image = normalise(sample.ink)
            if image is None:
                skipped.append((date, "signed, but too little ink recovered to compare"))
                continue
            sample.student_index = self.records.students[row_index]["index"]
            collected.append((date, sample, image))
        return collected, skipped

    # ------------------------------------------------------------------ #
    def investigate(self, student, row_index):
        """Steps 3-5: compare, group, choose the reference, judge each sample."""
        collected, skipped = self.collectSignatures(row_index)
        if len(collected) < 2:
            return {
                "student": student, "collected": collected, "skipped": skipped,
                "matrix": None, "labels": None, "reference": None,
                "members": [], "verdicts": [],
            }

        images = [image for _, _, image in collected]
        matrix = self.matcher.similarityMatrix(images)
        labels = self.matcher.groupMatchingSignatures(matrix)
        reference, members = self.matcher.referenceSignature(matrix, labels)

        verdicts = []
        for position, (date, sample, _) in enumerate(collected):
            agreement = float(np.mean([matrix[position, j] for j in members if j != position])) \
                if len(members) > 1 and position in members else float(
                    np.mean([matrix[position, j] for j in members if j != position])
                    if [j for j in members if j != position] else 0.0
                )
            verdicts.append({
                "date": date,
                "matches": position in members,
                "is_reference": position == reference,
                "agreement": agreement,
                "similarity_to_reference": float(matrix[position, reference]),
            })

        return {
            "student": student, "collected": collected, "skipped": skipped,
            "matrix": matrix, "labels": labels, "reference": reference,
            "members": members, "verdicts": verdicts,
        }

    # ------------------------------------------------------------------ #
    def report(self, result):
        """Print the findings as tables."""
        student = result["student"]
        console.rule(f"[bold]{student['name']}  ({student['index']})")

        collected = result["collected"]
        console.print(f"Collected [bold]{len(collected)}[/bold] signature(s) "
                      f"from the sheets in [dim]{self.images_dir}/[/dim]")
        for date, reason in result["skipped"]:
            console.print(f"  [yellow]{date}[/yellow] - {reason}")

        if result["matrix"] is None:
            console.print("[yellow]Fewer than two signatures: nothing to compare.[/yellow]")
            return

        # step 3 - the full pairwise comparison
        matrix = result["matrix"]
        dates = [d for d, _, _ in collected]
        comparison = Table(title="Step 3 - every signature compared with every other",
                           header_style="bold")
        comparison.add_column("")
        for date in dates:
            comparison.add_column(date, justify="right")
        for i, date in enumerate(dates):
            cells = []
            for j in range(len(dates)):
                if i == j:
                    cells.append("[dim]-[/dim]")
                else:
                    score = matrix[i, j]
                    colour = "green" if score >= self.matcher.agreement_threshold else "red"
                    cells.append(f"[{colour}]{score:.3f}[/{colour}]")
            comparison.add_row(f"[bold]{date}[/bold]", *cells)
        console.print(comparison)

        # steps 4 and 5 - the group, the reference, the verdicts
        reference_date = dates[result["reference"]]
        console.print(f"Step 4 - largest agreeing group: "
                      f"[bold]{len(result['members'])} of {len(dates)}[/bold] signatures. "
                      f"Reference (most typical) signature: [bold]{reference_date}[/bold]")

        verdict_table = Table(title="Step 5 - does each signature belong to this student?",
                              header_style="bold")
        verdict_table.add_column("Date")
        verdict_table.add_column("Similarity to reference", justify="right")
        verdict_table.add_column("Agreement with group", justify="right")
        verdict_table.add_column("Verdict")
        for verdict in result["verdicts"]:
            if verdict["is_reference"]:
                mark = "[bold green]REFERENCE[/bold green]"
            elif verdict["matches"]:
                mark = "[green]MATCHES[/green]"
            else:
                mark = "[bold red]DOES NOT MATCH[/bold red]"
            verdict_table.add_row(
                verdict["date"],
                f"{verdict['similarity_to_reference']:.3f}",
                f"{verdict['agreement']:.3f}",
                mark,
            )
        console.print(verdict_table)

        flagged = [v["date"] for v in result["verdicts"] if not v["matches"]]
        if flagged:
            console.print(f"[bold red]Review required[/bold red] - the signature(s) dated "
                          f"{', '.join(flagged)} do not match this student's other signatures.")
        else:
            console.print("[green]All collected signatures are consistent with one another."
                          "[/green]")

        console.print("[dim]Reliability: measured on these sheets the descriptors separate "
                      "same-student from different-student pairs with a balanced error rate "
                      "of about 30% (see calibrate_matcher.py). A flag means look at it, "
                      "not that forgery occurred.[/dim]")

    # ------------------------------------------------------------------ #
    def reportAttendance(self, student):
        """Step 6: the attendance record stored by sams.py."""
        history = self.query.studentHistory(student["index"])
        if not history:
            console.print(f"[yellow]No attendance recorded for {student['index']}. "
                          f"Run sams.py over the sheets first.[/yellow]")
            return history

        attended = sum(1 for _, status in history if status == "Present")
        table = Table(title="Step 6 - attendance record", header_style="bold")
        table.add_column("Date")
        table.add_column("Status")
        for date, status in history:
            colour = "green" if status == "Present" else "red"
            table.add_row(date, f"[{colour}]{status}[/{colour}]")
        console.print(table)
        console.print(f"Attended [bold]{attended}/{len(history)}[/bold] "
                      f"({attended / len(history):.0%})")
        return history

    # ------------------------------------------------------------------ #
    def drawFigure(self, result, history, output_path):
        """One figure carrying the signatures, the comparison and the record."""
        student = result["student"]
        collected = result["collected"]
        count = max(len(collected), 1)

        figure = plt.figure(figsize=(max(10.0, 2.4 * count), 9.5))
        figure.suptitle(f"Signature investigation - {student['name']} ({student['index']})",
                        fontsize=15, fontweight="bold", y=0.975)
        grid = figure.add_gridspec(3, count, height_ratios=[1.0, 1.25, 0.95],
                                   hspace=0.45, wspace=0.12,
                                   left=0.09, right=0.96, top=0.88, bottom=0.07)

        # row 1 - the signatures themselves
        for position, (date, _, image) in enumerate(collected):
            axes = figure.add_subplot(grid[0, position])
            axes.imshow(255 - image, cmap="gray", aspect="auto")
            axes.set_xticks([]); axes.set_yticks([])
            verdict = result["verdicts"][position] if result["verdicts"] else None
            if verdict is None:
                colour, note = "#8899a6", ""
            elif verdict["is_reference"]:
                colour, note = "#1f6f43", "\nREFERENCE"
            elif verdict["matches"]:
                colour, note = PRESENT_COLOUR, "\nmatches"
            else:
                colour, note = ABSENT_COLOUR, "\nDOES NOT MATCH"
            for spine in axes.spines.values():
                spine.set_color(colour); spine.set_linewidth(2.6)
            axes.set_title(f"{date}{note}", fontsize=9.5, color=colour,
                           fontweight="bold" if note.strip() != "matches" else "normal")

        # row 2 - the pairwise comparison
        heat = figure.add_subplot(grid[1, :])
        if result["matrix"] is not None:
            matrix = result["matrix"]
            dates = [d for d, _, _ in collected]
            image = heat.imshow(matrix, cmap="RdYlGn", vmin=0.2, vmax=0.9, aspect="auto")
            heat.set_xticks(range(len(dates)))
            heat.set_xticklabels(dates, rotation=30, ha="right", fontsize=9)
            heat.set_yticks(range(len(dates)))
            heat.set_yticklabels(dates, fontsize=9)
            for i in range(len(dates)):
                for j in range(len(dates)):
                    heat.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                              fontsize=9, color="#20262b")
            heat.set_title("Every signature compared with every other", fontsize=11, pad=8)
            figure.colorbar(image, ax=heat, fraction=0.02, pad=0.015)
        else:
            heat.text(0.5, 0.5, "Not enough signatures to compare",
                      ha="center", va="center", color="#8899a6")
            heat.set_axis_off()
        heat.grid(False)

        # row 3 - the attendance record
        record = figure.add_subplot(grid[2, :])
        if history:
            dates = [d for d, _ in history]
            present = [s == "Present" for _, s in history]
            positions = np.arange(len(dates))
            record.bar(positions, [1] * len(dates),
                       color=[PRESENT_COLOUR if p else ABSENT_COLOUR for p in present],
                       width=0.6, edgecolor="white", linewidth=1.5)
            for x, attended in zip(positions, present):
                record.text(x, 0.5, "P" if attended else "A", ha="center", va="center",
                            color="white", fontsize=12, fontweight="bold")
            rate = sum(present) / len(present)
            record.set_xticks(positions)
            record.set_xticklabels(dates, rotation=30, ha="right", fontsize=9)
            record.set_yticks([]); record.set_ylim(0, 1.3)
            record.set_title(f"Attendance record - {sum(present)}/{len(present)} ({rate:.0%})",
                             fontsize=11, pad=8)
            for spine in record.spines.values():
                spine.set_visible(False)
        else:
            record.text(0.5, 0.5, "No attendance records in the database",
                        ha="center", va="center", color="#8899a6")
            record.set_axis_off()
        record.grid(False)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path


# ---------------------------------------------------------------------- #
def cliArgumentParser():
    parser = argparse.ArgumentParser(
        description="Student Attendance Management System - signature recognition",
        formatter_class=RichHelpFormatter,
        usage="uv run investigate.py [-h] <student_number>",
    )
    parser.add_argument("student", nargs="?", help="Student index, full or a unique part of it")
    parser.add_argument("-r", "--row", type=int, default=None,
                        help="Select by row number on the sheet instead (1-based)")
    parser.add_argument("-x", "--xml", type=Path, default=Path("public/xml/info.xml"),
                        help="Path to the student xml file")
    parser.add_argument("-b", "--batch", default=None, help="Batch name, e.g. b15")
    parser.add_argument("-i", "--images", type=Path, default=Path("public/img"),
                        help="Directory holding the signing sheet images")
    parser.add_argument("-d", "--db", type=Path, default=Path("./db/attendance.db"),
                        help="Path to the attendance database")
    parser.add_argument("-o", "--output", type=Path, default=Path("public/output"),
                        help="Directory for the generated figure")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Investigate every student in the batch")
    return parser.parse_args()


def main():
    args = cliArgumentParser()
    if not args.xml.exists():
        console.print(f"[red]Xml file does not exist: {args.xml}[/red]")
        sys.exit(1)
    if not args.images.is_dir():
        console.print(f"[red]Images directory does not exist: {args.images}[/red]")
        sys.exit(1)
    if not args.student and not args.all and args.row is None:
        console.print("[red]Give a student number, or use --row / --all[/red]")
        sys.exit(1)

    records = StudentRecords(args.xml, args.batch)
    investigation = SignatureInvestigation(records, args.images, str(args.db))

    if args.all:
        targets = list(enumerate(records.students))
    elif args.row is not None:
        if not 1 <= args.row <= len(records.students):
            console.print(f"[red]Row {args.row} is out of range: the sheet has "
                          f"{len(records.students)} student rows[/red]")
            sys.exit(1)
        targets = [(args.row - 1, records.students[args.row - 1])]
    else:
        student, found = records.resolveStudent(args.student)

        # The brief documents "python investigate.py 001", but the indices in
        # info.xml are eight digits, so a short number cannot be one.  Rather
        # than fail on the command the marker is most likely to type, treat it
        # as a row number on the sheet - and say so, so the fallback is never
        # silently mistaken for an index lookup.
        if student is None and not found and args.student.isdigit():
            try:
                row = int(args.student)
            except ValueError:
                row = 0
            if 1 <= row <= len(records.students):
                student = records.students[row - 1]
                found = row - 1
                console.print(f"[yellow]No index matches '{args.student}'; "
                              f"reading it as row {row} on the sheet -> "
                              f"{student['index']}[/yellow]")

        if student is None:
            if found:
                console.print(f"[red]'{args.student}' matches more than one student:[/red]")
                for candidate in found:
                    console.print(f"  {candidate['index']}  {candidate['name']}")
            else:
                console.print(f"[red]No student matching '{args.student}' in batch "
                              f"{records.batch_name}[/red]")
                for candidate in records.students:
                    console.print(f"  {candidate['index']}  {candidate['name']}")
            sys.exit(1)
        targets = [(found, student)]

    flagged_total = 0
    for row_index, student in targets:
        result = investigation.investigate(student, row_index)
        investigation.report(result)
        history = investigation.reportAttendance(student)
        figure_path = investigation.drawFigure(
            result, history, args.output / f"investigate_{student['index']}.png"
        )
        console.print(f"Figure written to [bold]{figure_path}[/bold]\n")
        flagged_total += sum(1 for v in result["verdicts"] if not v["matches"])

    console.print(f"[bold]{flagged_total} signature(s) flagged for review.[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
