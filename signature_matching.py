import itertools

import cv2
import numpy as np
from skimage.feature import hog
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

NORMALISED_SIZE = (256, 128)          # (width, height)


WEIGHT_DIRECTION = 0.50
WEIGHT_HOG = 0.20
WEIGHT_HU = 0.00
WEIGHT_CHAMFER = 0.30

AGREEMENT_THRESHOLD = 0.50


def trimTails(ink, keep=0.16, bridge=14):
    """Discard a long thin flourish, keep the dense body of the signature.

    Many signers finish with a horizontal stroke several times longer than the
    signature itself.  Scaling the full bounding box into a fixed frame would
    then shrink the informative glyph into a corner while the featureless tail
    fills the frame, which destroys the comparison.  Columns are kept when
    their ink count reaches ``keep`` of the densest column - a tail is one or
    two pixels tall, the glyph is tens - and the longest surviving run is
    returned, bridging small gaps so a dotted stroke does not split it.
    """
    columns = (ink > 0).sum(axis=0).astype(float)
    if columns.max() == 0:
        return ink
    dense = np.where(columns >= keep * columns.max())[0]
    if dense.size == 0:
        return ink

    runs = []
    start = previous = dense[0]
    for position in dense[1:]:
        if position - previous > bridge:
            runs.append((start, previous))
            start = position
        previous = position
    runs.append((start, previous))

    first, last = max(runs, key=lambda r: int((ink[:, r[0]:r[1] + 1] > 0).sum()))
    return ink[:, first:last + 1]


def normalise(ink, size=NORMALISED_SIZE):
    """Trim, tightly crop, scale preserving aspect, and centre in a fixed box.

    Returns ``None`` when the mask holds too little ink to be a signature.
    """
    ink = trimTails(ink)
    points = cv2.findNonZero(ink)
    if points is None or len(points) < 40:
        return None

    x, y, w, h = cv2.boundingRect(points)
    if w < 8 or h < 8:
        return None
    body = ink[y:y + h, x:x + w]

    target_w, target_h = size
    scale = min((target_w - 12) / w, (target_h - 12) / h)
    resized = cv2.resize(
        body, (max(int(w * scale), 1), max(int(h * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    _, resized = cv2.threshold(resized, 40, 255, cv2.THRESH_BINARY)

    canvas = np.zeros((target_h, target_w), np.uint8)
    y0 = (target_h - resized.shape[0]) // 2
    x0 = (target_w - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


class SignatureMatcher:
    """Compares signatures and groups the ones that agree."""

    def __init__(self, agreement_threshold=AGREEMENT_THRESHOLD):
        self.agreement_threshold = agreement_threshold

    @staticmethod
    def directionDescriptor(image, bins=12, grid_x=4, grid_y=2):
        """Gradient-orientation histograms over a coarse spatial grid."""
        smoothed = cv2.GaussianBlur(image.astype(np.float32) / 255.0, (0, 0), 1.5)
        dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=5)
        dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=5)
        magnitude = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx) % np.pi

        height, width = smoothed.shape
        blocks = []
        for row in range(grid_y):
            for column in range(grid_x):
                y0, y1 = row * height // grid_y, (row + 1) * height // grid_y
                x0, x1 = column * width // grid_x, (column + 1) * width // grid_x
                histogram, _ = np.histogram(
                    angle[y0:y1, x0:x1], bins=bins, range=(0.0, np.pi),
                    weights=magnitude[y0:y1, x0:x1],
                )
                blocks.append(histogram)
        vector = np.concatenate(blocks)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def hogDescriptor(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        return hog(
            blurred,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

    @staticmethod
    def huDescriptor(image):
        """Log-scaled Hu moments; the log compresses their dynamic range."""
        moments = cv2.moments(image, binaryImage=True)
        hu = cv2.HuMoments(moments).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(hu != 0, -np.sign(hu) * np.log10(np.abs(hu)), 0.0)
        return np.nan_to_num(scaled)

    @staticmethod
    def cosine(a, b):
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))

    @staticmethod
    def huSimilarity(a, b):
        """Exponential falloff on Hu-moment distance, mapped onto 0..1."""
        return float(np.exp(-float(np.linalg.norm(a - b)) / 3.0))

    @staticmethod
    def alignedChamfer(first, second, shifts=(-10, -5, 0, 5, 10), angles=(-6, -3, 0, 3, 6)):
        """Best chamfer similarity over a small search of shifts and rotations.

        Chamfer distance measures how far each inked pixel of one image lies
        from the nearest inked pixel of the other.  Comparing without searching
        would punish two identical signatures that merely sit a few pixels
        apart after normalisation, so the second image is nudged and rotated
        over a small grid and the best agreement is kept.
        """
        if first.max() == 0 or second.max() == 0:
            return 0.0
        distance_first = cv2.distanceTransform(
            255 - (first > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3
        )
        points_first = first > 0
        height, width = second.shape
        best = 1e9

        for angle in angles:
            if angle:
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                rotated = cv2.warpAffine(second, matrix, (width, height))
            else:
                rotated = second
            distance_rotated = cv2.distanceTransform(
                255 - (rotated > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3
            )
            for dx in shifts:
                for dy in shifts:
                    translate = np.float32([[1, 0, dx], [0, 1, dy]])
                    moved = cv2.warpAffine(rotated, translate, (width, height))
                    moved_points = moved > 0
                    if not moved_points.any():
                        continue
                    moved_distance = cv2.warpAffine(
                        distance_rotated, translate, (width, height),
                        borderValue=float(distance_rotated.max()),
                    )
                    forward = float(moved_distance[points_first].mean())
                    backward = float(distance_first[moved_points].mean())
                    best = min(best, 0.5 * (forward + backward))

        return float(np.exp(-best / 14.0))

    def pairSimilarity(self, first, second):
        """Combined similarity plus its four components, all on 0..1."""
        direction = self.cosine(
            self.directionDescriptor(first), self.directionDescriptor(second)
        )
        hog = self.cosine(self.hogDescriptor(first), self.hogDescriptor(second))
        hu = self.huSimilarity(self.huDescriptor(first), self.huDescriptor(second))
        chamfer = self.alignedChamfer(first, second)
        combined = (
            WEIGHT_DIRECTION * direction
            + WEIGHT_HOG * hog
            + WEIGHT_HU * hu
            + WEIGHT_CHAMFER * chamfer
        )
        return combined, {"direction": direction, "hog": hog, "hu": hu, "chamfer": chamfer}

    # -------------------------------------------------------------- #
    def similarityMatrix(self, images):
        """Full pairwise similarity matrix for one student's signatures."""
        count = len(images)
        matrix = np.eye(count)
        for i, j in itertools.combinations(range(count), 2):
            score, _ = self.pairSimilarity(images[i], images[j])
            matrix[i, j] = matrix[j, i] = score
        return matrix

    def groupMatchingSignatures(self, matrix):
        """Cluster the samples so that mutually agreeing signatures group.

        Average-linkage agglomerative clustering is used on distance
        ``1 - similarity``.  Average linkage is chosen over single linkage
        because single linkage would chain two dissimilar samples together
        through any one intermediate that happened to resemble both.
        """
        count = matrix.shape[0]
        if count < 2:
            return np.zeros(count, dtype=int)

        distance = 1.0 - matrix
        np.fill_diagonal(distance, 0.0)
        distance = np.clip((distance + distance.T) / 2.0, 0.0, None)
        tree = linkage(squareform(distance, checks=False), method="average")
        return fcluster(tree, t=1.0 - self.agreement_threshold, criterion="distance")

    def referenceSignature(self, matrix, labels):
        """Index of the sample that best represents the student.

        The largest cluster is taken to be the student's true signature - a
        writer's habitual form should recur more often than any aberration -
        and within it the medoid, the sample most similar to all the others,
        is returned as the reference.  Ties are broken toward the tighter
        cluster, since a large loose cluster is weaker evidence than a smaller
        tight one.
        """
        best_cluster, best_key = None, None
        for cluster in set(labels):
            members = [i for i, label in enumerate(labels) if label == cluster]
            if len(members) < 2:
                cohesion = 0.0
            else:
                cohesion = float(np.mean([
                    matrix[a, b] for a, b in itertools.combinations(members, 2)
                ]))
            key = (len(members), cohesion)
            if best_key is None or key > best_key:
                best_cluster, best_key = cluster, key

        members = [i for i, label in enumerate(labels) if label == best_cluster]
        if len(members) == 1:
            return members[0], members
        medoid = max(members, key=lambda i: np.mean([matrix[i, j] for j in members if j != i]))
        return medoid, members
