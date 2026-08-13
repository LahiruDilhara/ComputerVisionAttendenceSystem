"""
attendance_visualizer.py

Part III (Visualization): renders a student's attendance record as charts.

A lecturer looking at an attendance record is really asking three separate
questions - *how often did this student attend*, *when did they miss*, and *is
that unusual for this class* - and no single chart answers all three.  A pie
chart saying "60%" hides whether the absences were scattered or consecutive,
and consecutive absences are the pattern that actually predicts a student
dropping out.  The summary is therefore a small dashboard of four coordinated
panels sharing one colour language: green for attended, red for missed.

Design decisions, each of which is defensible rather than decorative:

* **The donut carries the exact figure in its hole.**  Judging a quantity from
  the angle of a wedge is the least accurate of the common visual encodings, so
  the number does the precise work and the ring supplies the at-a-glance
  impression.
* **The session panel preserves order.**  Any summary statistic destroys the
  sequence, which is the most actionable part of the record.
* **The cumulative panel is drawn against the 80% requirement.**  A rate means
  nothing without the threshold it is judged by, so the threshold is drawn and
  the failing region shaded rather than left for the reader to work out.
* **The cohort panel supplies the comparison.**  "60%" is alarming if everyone
  else attends fully and unremarkable if the whole class is at 60%.  No
  single-student chart can answer that, so the class is drawn alongside.
* **Colour is never the only channel.**  Roughly 8% of men have some form of
  colour-vision deficiency, so every panel that uses green and red also encodes
  the same fact as a letter, a position, or a printed number.
* **Series are labelled directly.**  Values sit next to the marks they belong
  to instead of in a legend, so the reader never has to look away from the data
  to decode it.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # head-less: the marker may have no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PRESENT_COLOUR = "#2e8b57"
ABSENT_COLOUR = "#c0392b"
NEUTRAL_COLOUR = "#8899a6"
HIGHLIGHT_COLOUR = "#1f4e79"
GRID_COLOUR = "#dfe4e8"
REQUIRED_RATE = 0.80           # institutional attendance requirement


class StudentAttendance:
    """One student's record, with the derived figures the charts need."""

    def __init__(self, index, name, history):
        self.index = index
        self.name = name
        self.dates = [date for date, _ in history]
        self.present = [status == "Present" for _, status in history]

    @property
    def total(self):
        return len(self.present)

    @property
    def attended(self):
        return sum(self.present)

    @property
    def rate(self):
        return self.attended / self.total if self.total else 0.0

    def cumulativeRate(self):
        """Attendance rate after each successive session."""
        running = np.cumsum([1 if p else 0 for p in self.present])
        return list(running / np.arange(1, self.total + 1))

    def longestAbsentStreak(self):
        """Longest run of consecutive absences - the figure a pie chart hides."""
        best = current = 0
        for attended in self.present:
            current = 0 if attended else current + 1
            best = max(best, current)
        return best

    def meetsRequirement(self):
        return self.rate >= REQUIRED_RATE


class AttendanceVisualizer:
    """Builds the dashboard and the class-wide overview."""

    def __init__(self, style="seaborn-v0_8-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)

    # ------------------------------------------------------------------ #
    def studentDashboard(self, attendance, cohort=None, output_path=None):
        """Four-panel summary for one student.

        ``cohort`` is a list of ``(index, name, rate)`` used by the fourth
        panel only; pass ``None`` to draw the first three.
        """
        figure = plt.figure(figsize=(13, 8.5))
        figure.suptitle(
            f"Attendance summary  -  {attendance.name}  ({attendance.index})",
            fontsize=15, fontweight="bold", y=0.975,
        )
        grid = figure.add_gridspec(2, 2, hspace=0.45, wspace=0.26,
                                   left=0.07, right=0.96, top=0.88, bottom=0.10)

        self.panelDonut(figure.add_subplot(grid[0, 0]), attendance)
        self.panelSessions(figure.add_subplot(grid[0, 1]), attendance)
        self.panelCumulative(figure.add_subplot(grid[1, 0]), attendance)
        self.panelCohort(figure.add_subplot(grid[1, 1]), attendance, cohort)

        figure.legend(
            handles=[Patch(facecolor=PRESENT_COLOUR, label="Present"),
                     Patch(facecolor=ABSENT_COLOUR, label="Absent")],
            loc="lower center", ncol=2, frameon=False, fontsize=10,
        )
        return self._save(figure, output_path)

    # ------------------------------------------------------------------ #
    # panels                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def panelDonut(axes, attendance):
        """Overall rate.  A donut so the hole can hold the exact figure."""
        missed = attendance.total - attendance.attended
        values = [attendance.attended, missed] if missed else [attendance.attended]
        colours = [PRESENT_COLOUR, ABSENT_COLOUR] if missed else [PRESENT_COLOUR]

        axes.pie(values, colors=colours, startangle=90, counterclock=False,
                 wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2))
        axes.text(0, 0.10, f"{attendance.rate * 100:.0f}%", ha="center", va="center",
                  fontsize=27, fontweight="bold")
        axes.text(0, -0.22, f"{attendance.attended} of {attendance.total} sessions",
                  ha="center", va="center", fontsize=10, color="#55606a")
        axes.set_title("Overall attendance", fontsize=11.5, pad=12)
        axes.set_aspect("equal")

    @staticmethod
    def panelSessions(axes, attendance):
        """Per-session outcome in order - shows *when* absences happened."""
        positions = np.arange(attendance.total)
        colours = [PRESENT_COLOUR if p else ABSENT_COLOUR for p in attendance.present]

        axes.bar(positions, [1] * attendance.total, color=colours,
                 width=0.62, edgecolor="white", linewidth=1.5)
        for x, attended in zip(positions, attendance.present):
            axes.text(x, 0.5, "P" if attended else "A", ha="center", va="center",
                      color="white", fontsize=13, fontweight="bold")

        streak = attendance.longestAbsentStreak()
        note = (f"longest absent streak: {streak}" if streak
                else "no absences recorded")
        axes.text(0.5, 1.02, note, transform=axes.transAxes, ha="center",
                  fontsize=9, color="#55606a")

        axes.set_xticks(positions)
        axes.set_xticklabels(attendance.dates, rotation=45, ha="right", fontsize=9)
        axes.set_yticks([])
        axes.set_ylim(0, 1.25)
        axes.set_title("Session by session", fontsize=11.5, pad=26)
        axes.grid(False)
        for spine in axes.spines.values():
            spine.set_visible(False)

    @staticmethod
    def panelCumulative(axes, attendance):
        """Running rate against the requirement it is judged by."""
        positions = np.arange(1, attendance.total + 1)
        rates = attendance.cumulativeRate()

        axes.axhspan(0, REQUIRED_RATE, color=ABSENT_COLOUR, alpha=0.055)
        axes.axhline(REQUIRED_RATE, color=ABSENT_COLOUR, linestyle="--", linewidth=1.3)
        axes.text(0.02, REQUIRED_RATE + 0.03, f"{REQUIRED_RATE:.0%} requirement",
                  transform=axes.get_yaxis_transform(), fontsize=8.5,
                  color=ABSENT_COLOUR)

        axes.plot(positions, rates, color=HIGHLIGHT_COLOUR, linewidth=2.2, zorder=3)
        for x, rate, attended in zip(positions, rates, attendance.present):
            axes.plot(x, rate, marker="o", markersize=7, zorder=4,
                      color=PRESENT_COLOUR if attended else ABSENT_COLOUR)
        axes.annotate(f"{rates[-1]:.0%}", (positions[-1], rates[-1]),
                      textcoords="offset points", xytext=(8, 0), fontsize=9.5,
                      fontweight="bold", color=HIGHLIGHT_COLOUR, va="center")

        axes.set_ylim(-0.03, 1.10)
        axes.set_xlim(0.6, attendance.total + 0.7)
        axes.set_xticks(positions)
        axes.set_xticklabels(attendance.dates, rotation=45, ha="right", fontsize=9)
        axes.set_ylabel("Cumulative rate")
        axes.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axes.set_title("Attendance trend", fontsize=11.5, pad=12)
        axes.set_facecolor("white")
        axes.grid(color=GRID_COLOUR)

    @staticmethod
    def panelCohort(axes, attendance, cohort):
        """This student against every classmate, sorted."""
        if not cohort:
            axes.text(0.5, 0.5, "No cohort data available",
                      ha="center", va="center", color=NEUTRAL_COLOUR)
            axes.set_axis_off()
            return

        ordered = sorted(cohort, key=lambda row: row[2])
        labels = [name.split()[-1][:14] for _, name, _ in ordered]
        rates = [rate for _, _, rate in ordered]
        colours = [HIGHLIGHT_COLOUR if index == attendance.index else NEUTRAL_COLOUR
                   for index, _, _ in ordered]

        bars = axes.barh(np.arange(len(ordered)), rates, color=colours, height=0.66)
        mean_rate = float(np.mean(rates))
        axes.axvline(mean_rate, color="#d68910", linestyle="--", linewidth=1.3)
        axes.text(mean_rate, -0.92, f"class mean {mean_rate:.0%}", fontsize=8.5,
                  color="#b9770e", ha="center", va="center")

        for bar, rate in zip(bars, rates):
            axes.text(rate + 0.02, bar.get_y() + bar.get_height() / 2,
                      f"{rate:.0%}", va="center", fontsize=8.5, color="#41505c")

        axes.set_yticks(np.arange(len(ordered)))
        axes.set_yticklabels(labels, fontsize=9)
        axes.set_xlim(0, 1.18)
        axes.set_ylim(-1.4, len(ordered) - 0.4)
        axes.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axes.set_title("Compared with the class", fontsize=11.5, pad=12)
        axes.grid(color=GRID_COLOUR, axis="x")

    # ------------------------------------------------------------------ #
    def cohortOverview(self, sessions, output_path=None):
        """Headcount per session for the whole class.

        The enrolled figure is a single dashed reference line rather than a
        second bar series.  Two overlaid bar series need a legend, and a legend
        forces the reader to look away from the data to decode a colour; one
        labelled line states the same fact in place.
        """
        figure, axes = plt.subplots(figsize=(9, 4.6))
        dates = [row[0] for row in sessions]
        present = [row[1] for row in sessions]
        enrolled = max(row[2] for row in sessions)
        positions = np.arange(len(sessions))

        colours = [PRESENT_COLOUR if attended == enrolled else "#5f9e78"
                   for attended in present]
        axes.bar(positions, present, color=colours, width=0.6)
        for x, attended in zip(positions, present):
            axes.text(x, attended + 0.12, f"{attended}/{enrolled}", ha="center",
                      fontsize=9.5, fontweight="bold")

        axes.axhline(enrolled, color=NEUTRAL_COLOUR, linestyle="--", linewidth=1.3)
        axes.text(len(sessions) - 0.45, enrolled + 0.12, f"enrolled ({enrolled})",
                  fontsize=8.5, color="#55606a", ha="right")

        axes.set_xticks(positions)
        axes.set_xticklabels(dates, rotation=30, ha="right")
        axes.set_ylabel("Students attending")
        axes.set_ylim(0, enrolled + 1.0)
        axes.set_yticks(range(enrolled + 1))
        axes.set_title("Attendance by session", fontsize=12.5, fontweight="bold")
        figure.tight_layout()
        return self._save(figure, output_path)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _save(figure, output_path):
        if output_path is None:
            plt.close(figure)
            return None
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=150)
        plt.close(figure)
        return target
