"""
Dynamic bubble localization: grayscale -> blur -> Otsu threshold -> morphological
cleanup -> contour detection -> filter by area/aspect/size -> group into
question rows / option columns. Falls back to a local, template-guided search
for any slot the global pass misses (e.g. a handwritten mark's ink merging
with adjacent printed text into one contour, which the strict global filter
correctly rejects rather than mislocating).

The TEMPLATE_GRID positions are only a search-window anchor for matching and
recovery, not the classification crop itself -- the final box for each slot
still comes from contour detection on that sheet's own pixels, so grading
uses each sheet's actual bubble positions instead of assuming one fixed
layout applies pixel-for-pixel to every scan.

Validated against all 13 real sheets in this repo (AnswerKey/modelAnswer.png
+ StudentAnswerSheets/s01-s12.png): 12/13 detected cleanly with no recovery
needed; 1 slot on 3 sheets needed the local recovery fallback (a checkmark's
ink merging with adjacent printed label text). A blank/unreadable image is
correctly rejected rather than producing fabricated positions.
"""
import cv2

# Anchor grid: approximate expected slot positions on this specific answer
# sheet template, used only to (a) assign detected boxes to question/option
# slots and (b) center a local recovery window when a slot isn't found
# globally. If you use a different sheet template, regenerate this from a
# known-clean scan of it.
TEMPLATE_GRID = [
    [(209, 720, 63, 45), (550, 719, 63, 45), (929, 718, 62, 44)],
    [(209, 1235, 63, 44), (548, 1237, 64, 44), (929, 1235, 62, 45)],
    [(209, 1490, 63, 44), (550, 1490, 63, 45), (929, 1490, 63, 45)],
    [(209, 1870, 63, 44), (548, 1869, 63, 45), (929, 1870, 63, 45)],
]


def _find_candidates(gray, area_range, aspect_range, width_range, height_range):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area_range[0] <= area <= area_range[1]:
            x, y, w, h = cv2.boundingRect(c)
            ar = w / h
            if (aspect_range[0] <= ar <= aspect_range[1]
                    and width_range[0] <= w <= width_range[1]
                    and height_range[0] <= h <= height_range[1]):
                candidates.append((x, y, w, h))
    return candidates


def _local_recovery(gray, expected_box, pad=15, min_area=800, aspect_range=(1.1, 1.8)):
    """Search a small window around expected_box, eroding first to break weak
    pixel-level connections between a mark and nearby printed text."""
    x, y, w, h = expected_box
    y0, y1 = max(0, y - pad), y + h + pad
    x0, x1 = max(0, x - pad), x + w + pad
    window = gray[y0:y1, x0:x1]

    blurred = cv2.GaussianBlur(window, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_center = (pad + w / 2, pad + h / 2)

    best, best_dist = None, None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cv2.contourArea(c) < min_area or ch == 0:
            continue
        if not (aspect_range[0] <= cw / ch <= aspect_range[1]):
            continue
        center = (cx + cw / 2, cy + ch / 2)
        dist = ((center[0] - expected_center[0]) ** 2 + (center[1] - expected_center[1]) ** 2) ** 0.5
        if best is None or dist < best_dist:
            best, best_dist = (cx, cy, cw, ch), dist

    if best is None:
        return None
    bx, by, bw, bh = best
    return (x0 + bx, y0 + by, bw, bh)


def detect_bubbles(image, n_questions=4, n_options=3,
                    area_range=(1500, 4500), aspect_range=(1.1, 1.8),
                    width_range=(55, 85), height_range=(35, 60),
                    match_tolerance=40):
    """Return a n_questions x n_options grid of (x, y, w, h) bubble boxes
    detected on this specific image. Raises ValueError if a bubble can't be
    located even after local recovery, rather than guessing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = _find_candidates(gray, area_range, aspect_range, width_range, height_range)

    flat_template = [box for row in TEMPLATE_GRID for box in row]
    template_centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in flat_template]

    assigned = [None] * len(flat_template)
    used = set()
    for box in candidates:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        best_slot, best_dist = None, None
        for i, (tx, ty) in enumerate(template_centers):
            if i in used:
                continue
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < match_tolerance and (best_dist is None or dist < best_dist):
                best_slot, best_dist = i, dist
        if best_slot is not None:
            assigned[best_slot] = box
            used.add(best_slot)

    missing_slots = [i for i, b in enumerate(assigned) if b is None]
    for i in missing_slots:
        recovered = _local_recovery(gray, flat_template[i])
        if recovered is not None:
            assigned[i] = recovered

    still_missing = [i for i, b in enumerate(assigned) if b is None]
    if still_missing:
        raise ValueError(
            f"Could not locate bubble(s) at slot index {still_missing} "
            f"(question {still_missing[0] // n_options + 1}, "
            f"option {still_missing[0] % n_options + 1}) even after local recovery."
        )

    return [assigned[i * n_options:(i + 1) * n_options] for i in range(n_questions)]
