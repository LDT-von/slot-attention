"""Align explicit level-0 polygon annotations to ordered real patch tokens.

This pure geometry helper performs no I/O and never parses an unverified external
CSV format. It needs ``shapely`` at call time. Inputs and results may contain
patient-linked slide identifiers and must remain in private local evidence.

Caller-provided mapping attestations are validated structurally, but this
function cannot prove that a hash describes the files or that a token order
matches the actual model input: the caller must independently check those facts.
Unannotated and below-threshold patches are unknown, never normal controls.
"""

from collections.abc import Mapping, Sequence
import math
import re


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty exact identifier")
    return value


def _ring(points):
    points = _sequence(points, "polygon ring")
    if len(points) < 4:
        raise ValueError("polygon ring must have at least four positions")
    result = []
    for point in points:
        point = _sequence(point, "polygon position")
        if len(point) != 2:
            raise ValueError("polygon coordinates must be exactly two-dimensional")
        result.append((_number(point[0], "polygon x"), _number(point[1], "polygon y")))
    if result[0] != result[-1]:
        raise ValueError("GeoJSON polygon rings must be explicitly closed")
    return result


def _geometry(value, width, height, shape):
    value = _mapping(value, "geometry")
    kind = value.get("type")
    if kind not in ("Polygon", "MultiPolygon"):
        raise ValueError("only GeoJSON Polygon and MultiPolygon are supported")
    coordinates = _sequence(value.get("coordinates"), "coordinates")
    if not coordinates:
        raise ValueError("geometry must not be empty")
    groups = [coordinates] if kind == "Polygon" else coordinates
    polygons = []
    for group in groups:
        group = _sequence(group, "polygon rings")
        if not group:
            raise ValueError("polygon must contain a shell")
        polygons.append([_ring(ring) for ring in group])
    clean = {"type": kind, "coordinates": polygons[0] if kind == "Polygon" else polygons}
    try:
        geom = shape(clean)
    except Exception as exc:
        raise ValueError("invalid polygon geometry") from exc
    if geom.is_empty or not geom.is_valid or not math.isfinite(geom.area) or geom.area <= 0:
        raise ValueError("polygon geometry must be valid and have positive area")
    xmin, ymin, xmax, ymax = geom.bounds
    if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
        raise ValueError("polygon geometry extends beyond its level-0 slide bounds")
    return geom


def align_patch_annotations(patches, slides, polygons, *, thresholds=(0.25, 0.5, 0.75), expected_token_count=None):
    """Return labels and exact ROI-area fractions in explicit input token order.

    ``patches``: list of dicts with token_index (0..N-1), is_real (bool),
    slide_id, feature_row and box=[left, top, width, height]. Padding has the
    latter three as None. A real token on an unannotated slide may have box=None.

    ``slides``: mapping from exact slide ID to dict with positive feature_count,
    mapping_verified=True and a 64-hex mapping_sha256. Annotated slides also need
    positive level-0 width and height and coordinate_space='level0'. This metadata
    is an attestation, not verification of a source file's contents or token order.

    ``polygons``: mapping from exact slide ID to dict with
    coordinate_space='level0' and a nonempty ``geometries`` list of GeoJSON
    Polygon/MultiPolygon *geometries* (closed rings and optional holes). Unknown
    annotation schemas, including the uninspected BLCA CSV, must be adapted only
    after independent schema and coordinate-unit verification.

    The ≥0.5 mask is the primary mask. Sensitivity masks use each requested
    threshold. All real below-threshold or unannotated rows are 'unknown'; padding
    is separate. Fractions for an unannotated slide and padding are None.
    """
    try:
        from shapely.geometry import box as make_box, shape
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError("ROI alignment requires shapely") from exc

    patches = _sequence(patches, "patches")
    slides = _mapping(slides, "slides")
    polygons = _mapping(polygons, "polygons")
    thresholds = _sequence(thresholds, "thresholds")
    if not patches:
        raise ValueError("patches must not be empty")
    if expected_token_count is not None:
        expected = _integer(expected_token_count, "expected_token_count", minimum=1)
        if expected != len(patches):
            raise ValueError("token count differs from model attention")
    threshold_values = tuple(_number(t, "threshold") for t in thresholds)
    if not threshold_values or len(set(threshold_values)) != len(threshold_values):
        raise ValueError("thresholds must be nonempty and unique")
    if any(t <= 0 or t > 1 for t in threshold_values) or 0.5 not in threshold_values:
        raise ValueError("thresholds must be in (0, 1] and contain 0.5")

    slide_info = {}
    for slide_id, raw in slides.items():
        _identifier(slide_id, "slide_id")
        raw = _mapping(raw, "slide metadata")
        count = _integer(raw.get("feature_count"), "feature_count", minimum=1)
        if raw.get("mapping_verified") is not True:
            raise ValueError("each slide requires a confirmed feature-row mapping")
        digest = raw.get("mapping_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("each slide requires a 64-hex mapping_sha256")
        slide_info[slide_id] = {"feature_count": count, "mapping_sha256": digest.lower()}

    union_by_slide = {}
    for slide_id, raw in polygons.items():
        _identifier(slide_id, "annotation slide_id")
        if slide_id not in slides:
            raise ValueError("annotation slide is absent from the slide manifest")
        meta = slides[slide_id]
        if meta.get("coordinate_space") != "level0":
            raise ValueError("annotated slide metadata must declare level0")
        width = _number(meta.get("width"), "slide width")
        height = _number(meta.get("height"), "slide height")
        if width <= 0 or height <= 0:
            raise ValueError("slide dimensions must be positive")
        raw = _mapping(raw, "annotation")
        if raw.get("coordinate_space") != "level0":
            raise ValueError("annotation must explicitly declare level0")
        geometries = _sequence(raw.get("geometries"), "annotation geometries")
        if not geometries:
            raise ValueError("annotation geometries must not be empty")
        union = unary_union([_geometry(g, width, height, shape) for g in geometries])
        if union.is_empty or not union.is_valid or not math.isfinite(union.area) or union.area <= 0:
            raise ValueError("annotation union must be valid and nonempty")
        union_by_slide[slide_id] = union
        slide_info[slide_id].update({"width": width, "height": height})

    fractions = []
    real_mask = []
    statuses = []
    seen = set()
    for index, patch in enumerate(patches):
        patch = _mapping(patch, "patch")
        required = {"token_index", "is_real", "slide_id", "feature_row", "box"}
        if not required.issubset(patch):
            raise ValueError("each patch must explicitly contain all mapping fields")
        if _integer(patch.get("token_index"), "token_index") != index:
            raise ValueError("token_index must exactly follow input order")
        real = patch.get("is_real")
        if not isinstance(real, bool):
            raise ValueError("is_real must be boolean")
        real_mask.append(real)
        slide_id, feature_row, coords = patch.get("slide_id"), patch.get("feature_row"), patch.get("box")
        if not real:
            if any(value is not None for value in (slide_id, feature_row, coords)):
                raise ValueError("padding must not contain slide, feature row or coordinates")
            fractions.append(None)
            statuses.append("padding")
            continue
        _identifier(slide_id, "patch slide_id")
        if slide_id not in slide_info:
            raise ValueError("patch slide is absent from the slide manifest")
        feature_row = _integer(feature_row, "feature_row")
        if feature_row >= slide_info[slide_id]["feature_count"]:
            raise ValueError("feature_row exceeds slide feature_count")
        identity = (slide_id, feature_row)
        if identity in seen:
            raise ValueError("duplicate slide/feature_row in model input")
        seen.add(identity)
        if coords is None:
            if slide_id in union_by_slide:
                raise ValueError("annotated slides require coordinates for every input patch")
            fractions.append(None)
            statuses.append("unknown")
            continue
        coords = _sequence(coords, "patch box")
        if len(coords) != 4:
            raise ValueError("patch box must be [left, top, width, height]")
        x, y, w, h = (_number(v, "patch coordinate") for v in coords)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("patch bounds must be nonnegative with positive size")
        xmax, ymax, area = x + w, y + h, w * h
        if not all(math.isfinite(v) for v in (xmax, ymax, area)) or area <= 0 or xmax <= x or ymax <= y:
            raise ValueError("patch bounds or area cannot be represented safely")
        meta = slides[slide_id]
        if meta.get("coordinate_space") != "level0":
            raise ValueError("patch coordinates must explicitly declare level0")
        slide_width = _number(meta.get("width"), "slide width")
        slide_height = _number(meta.get("height"), "slide height")
        if slide_width <= 0 or slide_height <= 0 or xmax > slide_width or ymax > slide_height:
            raise ValueError("patch extends beyond its level-0 slide bounds")
        if slide_id not in union_by_slide:
            fractions.append(None)
            statuses.append("unknown")
            continue
        fraction = union_by_slide[slide_id].intersection(make_box(x, y, xmax, ymax)).area / area
        if not math.isfinite(fraction) or fraction < -1e-12 or fraction > 1 + 1e-12:
            raise ValueError("computed annotation area fraction is out of bounds")
        fraction = min(1.0, max(0.0, float(fraction)))
        fractions.append(fraction)
        statuses.append("tumor" if fraction >= 0.5 else "unknown")

    if not any(real_mask):
        raise ValueError("at least one real patch is required")
    sensitivity = {str(t): [bool(real and fraction is not None and fraction >= t)
                            for real, fraction in zip(real_mask, fractions)]
                   for t in threshold_values}
    return {
        "real_mask": real_mask,
        "tumor_mask": sensitivity["0.5"],
        "fractions": fractions,
        "status": statuses,
        "sensitivity": sensitivity,
        "counts": {"tokens": len(patches), "real": sum(real_mask),
                   "padding": len(patches) - sum(real_mask),
                   "tumor": sum(sensitivity["0.5"]), "unknown": statuses.count("unknown")},
        "mapping_attestations": slide_info,
        "mapping_evidence_independently_verified": False,
        "coordinate_space": "level0",
    }
