"""Read optional thumbnails and sparse patch tiles from one declared WSI.

This module does not establish slide identity or feature-row provenance. The
caller must independently verify that the supplied reader opens the exact slide
in the row manifest. No WSI is opened on import. Returned images and identifiers
are patient-linked and belong in ignored local_evidence.
"""

from collections.abc import Mapping, Sequence
from contextlib import closing
from numbers import Integral
from pathlib import Path
import re

import numpy as np


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _image_rgb(image):
    """Composite transparent background onto white, then return uint8 RGB."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise ValueError("slide reader must return PIL images")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def extract_slide_assets(reader, *, slide_id, slide_metadata, patches,
                         requested_rows, thumbnail_max_size=1024, tile_size=(224, 224)):
    """Return a full-slide thumbnail and requested, sparse level-0 tiles.

    ``patches`` is the ordered input manifest accepted by ROI alignment.
    ``requested_rows`` contains model token indices, not feature row numbers.
    Only rows whose declared slide ID exactly equals ``slide_id`` are read.
    Padding and other-slide rows are listed under ``missing_rows``. Coordinate
    boxes must be integer level-0 pixels, lie within reader dimensions, and have
    a valid feature row. Slide identity, manifest hashes and feature alignment
    remain independent caller attestations, not something this helper can prove.

    Returns ``thumbnail`` (RGB uint8), ``tiles`` (token-index -> RGB uint8),
    ``missing_rows`` (token-index -> reason), and metadata. Sparse tile sizes are
    normalized to ``tile_size`` for the existing gallery writer.
    """
    if not isinstance(slide_id, str) or not slide_id or slide_id.strip() != slide_id:
        raise ValueError("slide_id must be a nonempty exact identifier")
    if not isinstance(slide_metadata, Mapping):
        raise ValueError("slide_metadata must be a mapping")
    if slide_metadata.get("coordinate_space") != "level0":
        raise ValueError("slide metadata must declare level0 coordinates")
    if slide_metadata.get("mapping_verified") is not True:
        raise ValueError("feature-row mapping must be independently verified")
    digest = slide_metadata.get("mapping_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("mapping_sha256 must be a 64-character hex digest")
    count = _positive_int(slide_metadata.get("feature_count"), "feature_count")
    width = _positive_int(slide_metadata.get("width"), "slide width")
    height = _positive_int(slide_metadata.get("height"), "slide height")
    if tuple(getattr(reader, "dimensions", ())) != (width, height):
        raise ValueError("reader dimensions disagree with declared slide")
    if not isinstance(patches, Sequence) or isinstance(patches, (str, bytes)) or not patches:
        raise ValueError("patches must be a nonempty ordered sequence")
    if (not isinstance(requested_rows, (Sequence, np.ndarray))
            or isinstance(requested_rows, (str, bytes))
            or (isinstance(requested_rows, np.ndarray) and requested_rows.ndim != 1)):
        raise ValueError("requested_rows must be a sequence of token indices")
    if not isinstance(tile_size, (tuple, list)) or len(tile_size) != 2:
        raise ValueError("tile_size must be (width, height)")
    tile_width = _positive_int(tile_size[0], "tile width")
    tile_height = _positive_int(tile_size[1], "tile height")
    max_thumb = _positive_int(thumbnail_max_size, "thumbnail_max_size")
    if max_thumb > 8192 or max(tile_width, tile_height) > 4096:
        raise ValueError("requested output image size is too large")
    indices = []
    for index in requested_rows:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index < len(patches):
            raise ValueError("requested row index is outside the input manifest")
        index = int(index)
        if index in indices:
            raise ValueError("requested row indices must be unique")
        indices.append(index)
    # Validate every manifest index before any image read: order is essential.
    for index, row in enumerate(patches):
        if not isinstance(row, Mapping) or row.get("token_index") != index or isinstance(row.get("token_index"), bool):
            raise ValueError("patches must be in exact model token order")
        if not isinstance(row.get("is_real"), bool):
            raise ValueError("each row must specify boolean is_real")

    chosen, missing = {}, {}
    for index in indices:
        row = patches[index]
        if not row["is_real"]:
            missing[index] = "padding"
            continue
        if row.get("slide_id") != slide_id:
            missing[index] = "different_slide"
            continue
        feature_row = row.get("feature_row")
        if isinstance(feature_row, bool) or not isinstance(feature_row, int) or not 0 <= feature_row < count:
            raise ValueError("feature row is outside the declared feature table")
        box = row.get("box")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("requested matching row requires a level-0 xywh box")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in box):
            raise ValueError("patch boxes must use integer level-0 pixels")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError("requested patch extends outside slide dimensions")
        # Limit native decoder allocation for erroneous manifests.
        if w * h > 100_000_000:
            raise ValueError("requested patch exceeds native pixel limit")
        chosen[index] = (x, y, w, h)

    thumbnail = _image_rgb(reader.get_thumbnail((max_thumb, max_thumb)))
    tw, th = thumbnail.size
    if tw < 1 or th < 1 or max(tw, th) > max_thumb:
        raise ValueError("reader returned an invalid thumbnail size")
    if abs(tw * height - th * width) > max(width, height):
        raise ValueError("reader thumbnail aspect ratio disagrees with slide")
    tiles = {}
    for index, (x, y, w, h) in chosen.items():
        region = _image_rgb(reader.read_region((x, y), 0, (w, h)))
        if region.size != (w, h):
            raise ValueError("reader returned a region with unexpected dimensions")
        from PIL import Image
        tiles[index] = np.asarray(region.resize((tile_width, tile_height), Image.Resampling.LANCZOS))
    return {"thumbnail": np.asarray(thumbnail), "tiles": tiles,
            "missing_rows": missing, "slide_size": (width, height),
            "mapping_sha256": digest.lower(),
            "identity_independently_verified_by_this_helper": False}


def open_slide_assets(path, **kwargs):
    """Open and close a WSI with OpenSlide; no file access occurs until called."""
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("opening WSI files requires openslide-python and OpenSlide") from exc
    with closing(openslide.OpenSlide(str(Path(path)))) as reader:
        return extract_slide_assets(reader, **kwargs)
