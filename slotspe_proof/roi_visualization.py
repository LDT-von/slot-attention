"""Pure plotting helpers for annotated SlotSPE patch-reading diagnostics.

All spatial coordinates are level-0 slide pixels. No input files are read here;
the caller must establish the feature-row/slide/coordinate provenance first.
All figure writes are restricted to a *new* directory under ``local_evidence``.
"""

from collections import Counter
from pathlib import Path

import numpy as np


def _figure(width, height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def _image(value, name="thumbnail"):
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must have shape [height, width, 3 or 4]")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError(f"{name} must contain finite numeric RGB(A) values")
    if image.dtype.kind == "f":
        if np.any((image < 0) | (image > 1)):
            raise ValueError(f"floating-point {name} must be in [0, 1]")
    elif np.any((image < 0) | (image > 255)):
        raise ValueError(f"integer {name} must be in [0, 255]")
    return image


def _slide_size(value):
    size = np.asarray(value, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("slide_size must be finite positive (width, height) in level-0 pixels")
    return float(size[0]), float(size[1])


def _full_slide_thumbnail(image, slide_size):
    """Reject cropped/stretched thumbnails when full-slide extent is claimed.

    Thumbnail dimensions can differ by at most one thumbnail pixel from the
    level-0 aspect ratio because image resizing may round either dimension.
    Matching aspect ratios do not establish slide identity or prove that a
    thumbnail is uncropped: the caller must verify provenance independently.
    """
    width, height = slide_size
    thumb_height, thumb_width = image.shape[:2]
    if abs(thumb_width * height - thumb_height * width) > max(width, height):
        raise ValueError("thumbnail aspect ratio disagrees with full level-0 slide extent")


def _mask(value, length, name):
    mask = np.asarray(value)
    if mask.shape != (length,) or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array of length {length}")
    return mask


def _geometry(slide_size, boxes, visible_mask, real_mask):
    width, height = _slide_size(slide_size)
    rects = np.asarray(boxes, dtype=float)
    if rects.ndim != 2 or rects.shape[1] != 4:
        raise ValueError("boxes must have shape [tokens, 4] as (x, y, width, height)")
    visible = _mask(visible_mask, len(rects), "visible_mask")
    real = _mask(real_mask, len(rects), "real_mask")
    if np.any(visible & ~real):
        raise ValueError("visible_mask must exclude padding")
    shown = rects[visible]
    if not np.isfinite(shown).all():
        raise ValueError("visible patch boxes must be finite")
    if shown.size and (
        np.any(shown[:, :2] < 0) or np.any(shown[:, 2:] <= 0)
        or np.any(shown[:, 0] + shown[:, 2] > width)
        or np.any(shown[:, 1] + shown[:, 3] > height)
    ):
        raise ValueError("visible patch boxes must lie inside the level-0 slide")
    return (width, height), rects, visible, real


def _polygons(polygons, slide_size):
    width, height = slide_size
    result = []
    for ring in polygons:
        points = np.asarray(ring, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("each ROI outline needs at least three (x, y) points")
        if (not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points[:, 0] > width) or np.any(points[:, 1] > height)):
            raise ValueError("ROI outline must be finite and inside the level-0 slide")
        # Reject degenerate outlines; independently validated polygon topology
        # and exact slide provenance remain the caller's responsibility.
        signed_twice_area = np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
        if not np.isfinite(signed_twice_area) or abs(signed_twice_area) <= np.finfo(float).eps:
            raise ValueError("ROI outline must enclose a nonzero area")
        result.append(points)
    return result


def _outlines(ax, polygons):
    for points in polygons:
        closed = np.vstack((points, points[0]))
        ax.plot(closed[:, 0], closed[:, 1], color="#d7191c", linewidth=0.9)


def _axes_layout(ax, slide_size):
    width, height = slide_size
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_roi_overlay(thumbnail, slide_size, polygons, boxes, visible_mask, real_mask):
    """Return ``(figure, metadata)`` showing the slide and ROI outlines.

    ``visible_mask`` identifies rows of real *model input* on this exact slide.
    It does not define the pathological label of a patch.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    picture = _image(thumbnail)
    size, rects, visible, _ = _geometry(slide_size, boxes, visible_mask, real_mask)
    _full_slide_thumbnail(picture, size)
    rings = _polygons(polygons, size)
    fig = _figure(7, 6)
    ax = fig.subplots()
    ax.imshow(picture, extent=(0, size[0], size[1], 0), interpolation="nearest")
    outlines = [Rectangle((x, y), width, height) for x, y, width, height in rects[visible]]
    if outlines:
        ax.add_collection(PatchCollection(outlines, facecolor="none", edgecolor="#00b5d8",
                                          linewidth=0.35, alpha=0.7))
    _outlines(ax, rings)
    _axes_layout(ax, size)
    ax.set_title("ROI outlines (red); input patch coverage (cyan)")
    return fig, {"status": "ok", "visible_real_patch_count": int(visible.sum()),
                 "roi_outline_count": len(rings), "coordinates": "level0_xywh"}


def _rasterize(rects, visible, weights, slide_size, raster_shape):
    """Pixel-centre rasterization; overlapping patches are averaged, never summed."""
    height, width = raster_shape
    slide_width, slide_height = slide_size
    selected = rects[visible]
    x0 = np.ceil(selected[:, 0] * width / slide_width - 0.5).astype(np.int64)
    x1 = np.ceil((selected[:, 0] + selected[:, 2]) * width / slide_width - 0.5).astype(np.int64)
    y0 = np.ceil(selected[:, 1] * height / slide_height - 0.5).astype(np.int64)
    y1 = np.ceil((selected[:, 1] + selected[:, 3]) * height / slide_height - 0.5).astype(np.int64)
    x0, x1 = np.clip(x0, 0, width), np.clip(x1, 0, width)
    y0, y1 = np.clip(y0, 0, height), np.clip(y1, 0, height)
    has_pixel = (x0 < x1) & (y0 < y1)
    x0, x1, y0, y1 = (a[has_pixel] for a in (x0, x1, y0, y1))
    local_weights = weights[:, visible][:, has_pixel]
    counts = np.zeros((height + 1, width + 1), dtype=np.int64)
    totals = np.zeros((weights.shape[0], height + 1, width + 1), dtype=float)
    for yy, xx, sign in ((y0, x0, 1), (y0, x1, -1), (y1, x0, -1), (y1, x1, 1)):
        np.add.at(counts, (yy, xx), sign)
        for slot in range(weights.shape[0]):
            np.add.at(totals[slot], (yy, xx), sign * local_weights[slot])
    counts = counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    totals = totals.cumsum(axis=1).cumsum(axis=2)[:, :-1, :-1]
    covered = counts > 0
    result = np.full(totals.shape, np.nan)
    np.divide(totals, counts[None], out=result, where=covered[None])
    return result, int((~has_pixel).sum())


def render_slot_heatmaps(thumbnail, slide_size, polygons, boxes, weights, visible_mask, real_mask):
    """Plot per-slot weights conditioned on all real input rows with one scale.

    ``weights`` is [slots, all input rows], already renormalized over all real
    rows, including those from other slides. Padded rows are never displayed.
    Missing pixels are blank; overlapping patch values are averaged per pixel.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    picture = _image(thumbnail)
    size, rects, visible, real = _geometry(slide_size, boxes, visible_mask, real_mask)
    _full_slide_thumbnail(picture, size)
    rings = _polygons(polygons, size)
    data = np.asarray(weights, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(rects) or data.shape[0] < 1:
        raise ValueError("weights must have shape [slots, input rows]")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("weights must be finite and nonnegative")
    if not real.any() or not np.allclose(data[:, real].sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("weights must be conditioned to sum to one on all real rows")
    if not visible.any():
        raise ValueError("no input patch has confirmed coordinates on the selected slide")
    if picture.shape[0] * picture.shape[1] * data.shape[0] > 16_000_000:
        raise ValueError("thumbnail resolution is too large for this raster; provide a smaller thumbnail")
    pixels, too_small = _rasterize(rects, visible, data, size, picture.shape[:2])
    if not np.isfinite(pixels).any():
        raise ValueError("no visible patch covers a thumbnail pixel centre; use a higher-resolution thumbnail")
    vmax = float(data[:, real].max())
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("white")
    slots = data.shape[0]
    columns = min(4, slots)
    rows = (slots + columns - 1) // columns
    fig = _figure(4 * columns, 3.8 * rows)
    axes = fig.subplots(rows, columns, squeeze=False)
    artists = []
    for slot, ax in enumerate(axes.flat):
        if slot >= slots:
            ax.set_visible(False)
            continue
        artist = ax.imshow(np.ma.masked_invalid(pixels[slot]), cmap=cmap, norm=norm,
                           extent=(0, size[0], size[1], 0), interpolation="nearest")
        artists.append(artist)
        _outlines(ax, rings)
        _axes_layout(ax, size)
        ax.set_title(f"Slot {slot + 1}")
    fig.colorbar(artists[0], ax=[ax for ax in axes.flat if ax.get_visible()],
                 label="Attention conditioned on all real input patches")
    fig.suptitle("Blank = unobserved at this resolution; red = ROI outline")
    return fig, {"status": "ok", "slot_count": slots, "visible_real_patch_count": int(visible.sum()),
                 "real_patch_count": int(real.sum()), "unrepresented_subpixel_patches": too_small,
                 "shared_color_range": [0.0, vmax], "coordinate_space": "level0_xywh",
                 "overlap_rasterization": "mean_of_covering_patches",
                 "weights": "conditioned_on_all_real_input_rows"}


def render_overlap_matrix(overlap):
    """Plot a [slots, slots] pairwise reading-overlap matrix on [0, 1]."""
    matrix = np.asarray(overlap, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("overlap must be a nonempty square matrix")
    if (not np.isfinite(matrix).all() or np.any(matrix < -1e-8) or np.any(matrix > 1 + 1e-8)
            or not np.allclose(matrix, matrix.T, atol=1e-8)
            or not np.allclose(np.diag(matrix), 1.0, atol=1e-8)):
        raise ValueError("overlap must be finite, symmetric, within [0,1] and have unit diagonal")
    matrix = np.clip(matrix, 0.0, 1.0)  # permit numerical round-off only
    fig = _figure(7, 6)
    ax = fig.subplots()
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(matrix)), [str(i + 1) for i in range(len(matrix))])
    ax.set_yticks(range(len(matrix)), [str(i + 1) for i in range(len(matrix))])
    ax.set_xlabel("Slot")
    ax.set_ylabel("Slot")
    ax.set_title("Reading overlap; uniform slots also have overlap 1")
    fig.colorbar(image, ax=ax, label="Overlap")
    return fig, {"status": "ok", "slot_count": len(matrix), "color_range": [0.0, 1.0]}


def render_pathway_heatmap(weights, pathway_names, max_tick_labels=30):
    """Plot all slot-by-pathway token weights, preserving their given order.

    Long pathway lists thin axis *labels* only: no token columns are dropped.
    This depicts pathway-token reading, not individual gene attribution.
    """
    matrix = np.asarray(weights, dtype=float)
    names = list(pathway_names)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("weights must be a nonempty [slots, pathways] matrix")
    if len(names) != matrix.shape[1] or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("pathway_names must provide one nonempty name per token")
    if len(set(names)) != len(names):
        raise ValueError("pathway token names must be unique")
    if (not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)):
        raise ValueError("pathway weights must be finite, nonnegative, and normalized per slot")
    if isinstance(max_tick_labels, bool) or not isinstance(max_tick_labels, int) or max_tick_labels < 1:
        raise ValueError("max_tick_labels must be a positive integer")
    stride = max(1, (len(names) + max_tick_labels - 1) // max_tick_labels)
    ticks = list(range(0, len(names), stride))
    fig = _figure(min(20, max(7, len(names) * 0.18)), max(4, matrix.shape[0] * 0.45 + 3))
    ax = fig.subplots()
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0,
                      vmax=float(matrix.max()), interpolation="nearest")
    ax.set_yticks(range(matrix.shape[0]), [str(i + 1) for i in range(matrix.shape[0])])
    ax.set_xticks(ticks, [names[i] for i in ticks], rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Pathway tokens (all columns shown; some names omitted for legibility)")
    ax.set_ylabel("Slot")
    ax.set_title("Slot reading of pathway tokens; not single-gene attribution")
    fig.colorbar(image, ax=ax, label="Attention per pathway token")
    return fig, {"status": "ok", "slot_count": matrix.shape[0], "pathway_count": len(names),
                 "displayed_tick_count": len(ticks), "pathway_order_preserved": True,
                 "single_gene_attribution": False}


def render_top_gallery(tiles, top_indices, real_mask, tumor_mask=None):
    """Plot optional input-row tiles; ``None`` tiles return an explicit gap.

    Rows in ``top_indices`` are slot-specific selections of the same input bag;
    repeated row numbers identify exact cross-slot overlap. Non-tumor labels are
    rendered as 'unknown', not as histologically normal.
    """
    if tiles is None:
        return None, {"status": "missing_patch_tiles", "reason": "No patch images were supplied"}
    if isinstance(tiles, dict):
        real = np.asarray(real_mask)
        if real.ndim != 1 or real.dtype.kind != "b":
            raise ValueError("real_mask must be a 1-D boolean array")
        pictures = {}
        for key, tile in tiles.items():
            if (isinstance(key, bool) or not isinstance(key, (int, np.integer)) or
                    key < 0 or key >= len(real) or not real[key]):
                raise ValueError("sparse tile keys must refer to real input rows")
            pictures[int(key)] = _image(tile, "patch tile")
    else:
        data = np.asarray(tiles)
        if data.ndim != 4 or data.shape[0] < 1:
            raise ValueError("tiles must have shape [input rows, height, width, 3 or 4]")
        for tile in data:
            _image(tile, "patch tile")
        real = _mask(real_mask, len(data), "real_mask")
        pictures = {index: tile for index, tile in enumerate(data) if real[index]}
    top = np.asarray(top_indices)
    if top.ndim != 2 or min(top.shape) < 1 or top.dtype.kind not in "iu":
        raise ValueError("top_indices must be a nonempty integer [slots, K] matrix")
    if np.any(top < 0) or np.any(top >= len(real)) or not real[top].all():
        raise ValueError("top indices must refer to real input rows, never padding")
    if any(len(set(row.tolist())) != len(row) for row in top):
        raise ValueError("top indices may not repeat within one slot")
    if tumor_mask is None:
        tumor = np.zeros(len(real), dtype=bool)
    else:
        tumor = _mask(tumor_mask, len(real), "tumor_mask")
        if np.any(tumor & ~real):
            raise ValueError("padding cannot be labeled as tumor")
    frequency = Counter(int(i) for row in top for i in row)
    fig = _figure(max(4, top.shape[1] * 1.45), max(3, top.shape[0] * 1.6))
    axes = fig.subplots(top.shape[0], top.shape[1], squeeze=False)
    missing = 0
    for slot in range(top.shape[0]):
        for rank in range(top.shape[1]):
            index = int(top[slot, rank])
            ax = axes[slot, rank]
            if index in pictures:
                ax.imshow(pictures[index], interpolation="nearest")
            else:
                missing += 1
                ax.text(0.5, 0.5, "tile missing", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            label = "annotated tumor" if tumor[index] else "unknown"
            ax.set_title(f"row {index}; {label}\nseen in {frequency[index]} slot(s)", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if rank == 0:
                ax.set_ylabel(f"Slot {slot + 1}")
    fig.suptitle("Top patches; row numbers identify exact shared inputs")
    return fig, {"status": "partial_missing_patch_tiles" if missing else "ok", "slot_count": top.shape[0],
                 "missing_tile_panels": missing, "k": top.shape[1],
                 "distinct_shared_rows": sum(1 for count in frequency.values() if count > 1)}


def save_figures(figures, output_dir):
    """Save named figures to a new private directory; never overwrite outputs.

    Supported names are fixed so caller-controlled names cannot traverse paths.
    The target must resolve under this checkout's ignored ``local_evidence``.
    Missing optional figures are returned in the receipt and are not fabricated.
    """
    allowed = {"roi_overlay", "slot_heatmaps", "overlap_matrix", "top_gallery", "pathway_heatmap"}
    if not isinstance(figures, dict) or not figures or not set(figures).issubset(allowed):
        raise ValueError("figures must use supported fixed names")
    private = (Path(__file__).resolve().parents[1] / "local_evidence").resolve()
    target = Path(output_dir).resolve()
    if target == private or not target.is_relative_to(private):
        raise ValueError("output directory must be a new child of local_evidence")
    target.mkdir(parents=True, exist_ok=False)
    saved, missing = {}, []
    for name, fig in figures.items():
        if fig is None:
            missing.append(name)
            continue
        destination = target / f"{name}.png"
        fig.savefig(destination, dpi=150, facecolor="white")
        saved[name] = str(destination)
    return {"output_dir": str(target), "saved": saved, "missing": missing}
