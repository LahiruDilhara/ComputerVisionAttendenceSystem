

import itertools
from pathlib import Path

import numpy as np
from rich import print

from signature_extraction import SignatureExtractor
from signature_matching import (
    WEIGHT_CHAMFER, WEIGHT_DIRECTION, WEIGHT_HOG, WEIGHT_HU,
    SignatureMatcher, normalise,
)
from student_records import StudentRecords


def collectSamples(images_dir, records):
    """Every normalised signature on every sheet, tagged with its owner."""
    extractor = SignatureExtractor(expected_rows=len(records.students))
    matcher = SignatureMatcher()
    samples = []
    for path in sorted(Path(images_dir).glob("*.*")):
        date = records.dateFromFilename(path)
        if date is None:
            continue
        try:
            extracted = extractor.extract(path, date)
        except ValueError as error:
            print(f"[yellow]skipped {path.name}: {error}[/yellow]")
            continue
        for row, sample in enumerate(extracted):
            image = normalise(sample.ink)
            if image is None:
                continue
            samples.append((records.students[row]["index"], date, image, matcher))
    return samples


def main():
    records = StudentRecords("public/xml/info.xml")
    samples = collectSamples("public/img", records)
    matcher = SignatureMatcher()
    print(f"{len(samples)} usable signatures from {len(records.students)} students\n")

    genuine, impostor = [], []
    parts_genuine = {k: [] for k in ("direction", "hog", "hu", "chamfer")}
    parts_impostor = {k: [] for k in ("direction", "hog", "hu", "chamfer")}

    for (index_a, _, image_a, _), (index_b, _, image_b, _) in itertools.combinations(samples, 2):
        score, parts = matcher.pairSimilarity(image_a, image_b)
        if index_a == index_b:
            genuine.append(score)
            for k, v in parts.items():
                parts_genuine[k].append(v)
        else:
            impostor.append(score)
            for k, v in parts.items():
                parts_impostor[k].append(v)

    genuine, impostor = np.array(genuine), np.array(impostor)

    def dprime(a, b):
        return (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))

    print(f"[bold]Individual descriptors[/bold]")
    print(f"  {'descriptor':<12}{'same':>8}{'different':>11}{'d-prime':>10}")
    for key in ("direction", "hog", "hu", "chamfer"):
        a, b = np.array(parts_genuine[key]), np.array(parts_impostor[key])
        print(f"  {key:<12}{a.mean():>8.3f}{b.mean():>11.3f}{dprime(a, b):>10.2f}")

    weights = (f"{WEIGHT_DIRECTION} direction + {WEIGHT_HOG} hog + "
               f"{WEIGHT_HU} hu + {WEIGHT_CHAMFER} chamfer")
    print(f"\n[bold]Combined[/bold] ({weights})")
    print(f"  same-student pairs      n={genuine.size:<4} mean={genuine.mean():.3f} sd={genuine.std():.3f}")
    print(f"  different-student pairs n={impostor.size:<4} mean={impostor.mean():.3f} sd={impostor.std():.3f}")
    print(f"  discriminability d'     {dprime(genuine, impostor):.2f}")

    best = min(
        ((genuine < t).mean() * 0.5 + (impostor >= t).mean() * 0.5, t)
        for t in np.arange(0.10, 0.95, 0.005)
    )
    error, threshold = best
    print(f"\n[bold]Best operating point[/bold]")
    print(f"  threshold                {threshold:.3f}")
    print(f"  same pair called different (false reject) : {(genuine < threshold).mean():.1%}")
    print(f"  different pair called same (false accept) : {(impostor >= threshold).mean():.1%}")
    print(f"  balanced error rate                       : {error:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
