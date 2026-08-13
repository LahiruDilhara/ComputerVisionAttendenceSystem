"""
signature_extraction.py

Locates the six signature cells on a signing sheet and returns a clean ink
mask for each one.  This is the input stage for Part IV (Recognition).

Part II (sams.py) only needs to know *whether* a box holds ink, so it can work
on a coarse binary image.  Recognition needs the actual shape of the stroke, so
this module goes further in three ways:

1. **Illumination is flattened before thresholding.**  A hand-held photograph of
   paper has a bright patch and a hand shadow; on these sheets that difference
   is larger than the difference between paper and pale ink, so one global
   threshold cannot serve the whole page.
2. **Ink is separated by colour, not only by darkness.**  The form is printed in
   black (achromatic) while students sign in blue ballpoint (high saturation).
   Thresholding on HSV saturation lifts the handwriting off the form without
   any morphology, so a signature crossing a ruled line stays in one piece
   instead of being cut into fragments.
3. **Ink outside the cell is recovered.**  Signatures routinely overshoot the
   right border of their box.  Cropping to the printed cell would silently
   discard part of every signature, which is fatal when the shape is the thing
   being compared.

Cells are located with the same contour approach used in sams.py so that both
parts of the project agree about which box belongs to which student, but the
boxes are sorted explicitly by vertical position rather than relying on the
order OpenCV happens to return contours in.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

WORKING_WIDTH = 1600
MIN_COMPONENT_AREA = 60
INK_RATIO_THRESHOLD = 0.010


@dataclass
class SignatureSample:
    """One signature box on one sheet."""

    date: str
    row_index: int
    student_index: str = ""
    ink: np.ndarray = field(default=None, repr=False)      # binary, cell-sized
    colour: np.ndarray = field(default=None, repr=False)   # BGR crop
    ink_pixels: int = 0
    ink_ratio: float = 0.0
    present: bool = False

    @property
    def label(self):
        return f"{self.student_index}@{self.date}"


class SignatureExtractor:
    """Finds the signature cells on a sheet and extracts the handwriting."""

    def __init__(self, expected_rows=6, working_width=WORKING_WIDTH):
        self.expected_rows = expected_rows
        self.working_width = working_width

    # ------------------------------------------------------------------ #
    # preprocessing                                                       #
    # ------------------------------------------------------------------ #
    def loadImage(self, image_path):
        """Read and rescale a sheet to the fixed working width."""
        colour = cv2.imread(str(image_path))
        if colour is None:
            raise FileNotFoundError(f"cannot read image: {image_path}")
        height, width = colour.shape[:2]
        scale = self.working_width / width
        return cv2.resize(
            colour,
            (self.working_width, int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def flattenIllumination(gray):
        """Divide out the lighting field estimated by a wide median blur.

        The kernel is far wider than any pen stroke, so the median inside any
        window is the paper itself and the blurred copy is effectively a
        photograph of the page with nothing written on it.  Dividing by it
        cancels the lighting term and leaves ink on a uniform white background.
        """
        background = cv2.medianBlur(gray, 51)
        return cv2.divide(gray, background, scale=255)

    @staticmethod
    def binarize(flattened):
        """Adaptive Gaussian threshold; ink becomes white on black."""
        return cv2.adaptiveThreshold(
            flattened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 12,
        )

    @staticmethod
    def extractRules(binary, vertical_divisor=60):
        """Isolate the printed table rules by directional morphological opening.

        Opening with a long thin horizontal element survives only runs of ink
        at least that long - ruled lines - and erases letters and signature
        strokes.  The transposed pass recovers the vertical rules.  Kernel
        sizes are fractions of the image so they hold at any resolution.

        The vertical element must stay shorter than one table row, otherwise
        the short rules that divide adjacent cells are erased along with the
        handwriting and the six signature boxes never form as closed contours.
        At the working width a row is roughly 47 px tall, so the divisor is set
        to keep the element near 35 px.
        """
        height, width = binary.shape
        horizontal = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 25, 10), 1)),
        )
        vertical = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // vertical_divisor, 8))),
        )
        return horizontal, vertical

    # ------------------------------------------------------------------ #
    # cell location                                                       #
    # ------------------------------------------------------------------ #
    def findSignatureCells(self, binary):
        """Return the signature cells, ordered top to bottom.

        The signature column is the rightmost column of the largest table, and
        the student rows are the bottom-most cells in that column - everything
        above them is the column header and the lecturer block.

        sams.py reaches the same six cells by slicing the contour list and
        reversing it, which works only because OpenCV happens to return these
        contours bottom-to-top.  That is an implementation detail rather than a
        guarantee, and if it ever changed, every signature would be attributed
        to the wrong student silently.  Here the cells are sorted by vertical
        position explicitly.
        """
        horizontal, vertical = self.extractRules(binary)
        grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
        contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []

        boxes = []
        for index, contour in enumerate(contours):
            if hierarchy[0][index][3] == -1:          # keep inner cells only
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 25:
                boxes.append((x, y, w, h))
        if not boxes:
            return []

        right_edge = max(x + w for x, y, w, h in boxes)
        column = [b for b in boxes if (b[0] + b[2]) > right_edge - 50]
        if len(column) < self.expected_rows:
            return []

        column.sort(key=lambda b: b[1])               # top to bottom
        return column[-self.expected_rows:]           # bottom-most N are the students

    # ------------------------------------------------------------------ #
    # ink isolation                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def chromaticInk(colour):
        """Ink mask built from colour saturation alone.

        Printed rules and printed text are achromatic and sit at very low
        saturation; blue ballpoint sits high.  A saturation threshold therefore
        separates handwriting from the form without touching the rules at all,
        which is why a stroke crossing a rule survives intact.
        """
        hsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        mask = ((saturation > 70) & (value < 200)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    @staticmethod
    def achromaticInk(binary, rules):
        """Fallback channel for pens the chromatic test cannot see.

        A signature in true black is indistinguishable from the form by colour,
        so it has to be recovered by subtracting the rules.  They are dilated
        first because JPEG compression leaves them with soft edges that would
        otherwise survive as thin residue.
        """
        thick = cv2.dilate(rules, np.ones((7, 7), np.uint8))
        ink = cv2.bitwise_and(binary, cv2.bitwise_not(thick))
        return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def isolateHandwriting(self, colour, binary, rules, cells):
        """Union of both ink channels, restricted to the signature region.

        Neither channel is sufficient alone - the chromatic one is blind to a
        black pen, the achromatic one fragments strokes at the rules - so the
        union keeps the strengths of both and the size filter downstream
        removes the speckle it lets through.
        """
        ink = cv2.bitwise_or(self.chromaticInk(colour), self.achromaticInk(binary, rules))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        left = min(x for x, y, w, h in cells) + 4
        top = min(y for x, y, w, h in cells) - 6
        bottom = max(y + h for x, y, w, h in cells) + 40

        region = np.zeros_like(ink)
        region[max(top, 0):bottom, left:] = 255       # extend right for overflow
        return cv2.bitwise_and(ink, region)

    # ------------------------------------------------------------------ #
    def extract(self, image_path, date=""):
        """Full extraction: returns one SignatureSample per student row."""
        colour = self.loadImage(image_path)
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        binary = self.binarize(self.flattenIllumination(gray))
        horizontal, vertical = self.extractRules(binary)
        rules = cv2.bitwise_or(horizontal, vertical)

        cells = self.findSignatureCells(binary)
        if len(cells) != self.expected_rows:
            raise ValueError(
                f"{image_path}: found {len(cells)} signature cells, "
                f"expected {self.expected_rows}"
            )

        ink = self.isolateHandwriting(colour, binary, rules, cells)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

        bands = [(y, y + h) for x, y, w, h in cells]
        row_height = float(np.mean([b - a for a, b in bands]))
        owned = [[] for _ in bands]
        ink_pixels = [0] * len(bands)

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < MIN_COMPONENT_AREA:
                continue
            if width < 12 and height < 12:
                continue
            if height < 10 and width > 150:          # stray underline, not a signature
                continue

            ys = np.where(labels == label)[0]
            overlap = [int(((ys >= a) & (ys < b)).sum()) for a, b in bands]

            # A component no taller than about one and a half rows is one
            # signature crossing a rule, so it goes whole to the row it mostly
            # occupies.  A taller one is two signatures that touched and
            # merged, so each row is credited only with the ink inside it.
            if height <= 1.5 * row_height:
                winner = int(np.argmax(overlap))
                ink_pixels[winner] += area
                owned[winner].append(label)
            else:
                for index, share in enumerate(overlap):
                    if share:
                        ink_pixels[index] += share
                        owned[index].append(label)

        samples = []
        for index, (x, y, w, h) in enumerate(cells):
            mask = np.zeros_like(ink)
            for label in owned[index]:
                mask[labels == label] = 255
            top, bottom = bands[index]
            crop = mask[top:bottom + 20, x:]
            ratio = ink_pixels[index] / float(w * h) if w * h else 0.0
            samples.append(
                SignatureSample(
                    date=date,
                    row_index=index,
                    ink=crop,
                    colour=colour[top:bottom + 20, x:],
                    ink_pixels=ink_pixels[index],
                    ink_ratio=ratio,
                    present=ratio > INK_RATIO_THRESHOLD,
                )
            )
        return samples
