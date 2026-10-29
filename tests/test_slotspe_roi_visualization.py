"""Generated images only: no slide, patient, checkpoint or inference input."""

import io

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from slotspe_proof import roi_visualization as viz


@pytest.fixture
def synthetic():
    return {
        "thumbnail": np.full((4, 4, 3), 240, dtype=np.uint8),
        "slide_size": (4, 4),
        "polygons": [np.array([[0, 0], [2, 0], [2, 2], [0, 2]])],
        "boxes": np.array([[0, 0, 2, 2], [1, 1, 2, 2], [np.nan] * 4]),
        "visible": np.array([True, True, False]),
        "real": np.array([True, True, False]),
        "weights": np.tile([0.75, 0.25, 100.0], (8, 1)),
    }


def test_raster_preserves_unknown_and_averages_overlaps(synthetic):
    result, skipped = viz._rasterize(synthetic["boxes"], synthetic["visible"],
                                     synthetic["weights"], synthetic["slide_size"], (4, 4))
    assert skipped == 0
    assert result.shape == (8, 4, 4)
    assert result[0, 0, 0] == pytest.approx(0.75)
    assert result[0, 1, 1] == pytest.approx(0.50)
    assert result[0, 2, 2] == pytest.approx(0.25)
    assert np.isnan(result[0, 3, 3])


def test_figures_use_slide_coordinates_and_one_color_scale(synthetic):
    slide, info = viz.render_roi_overlay(
        synthetic["thumbnail"], synthetic["slide_size"], synthetic["polygons"],
        synthetic["boxes"], synthetic["visible"], synthetic["real"])
    assert info["visible_real_patch_count"] == 2
    np.testing.assert_allclose(slide.axes[0].lines[0].get_xdata(), [0, 2, 2, 0, 0])
    np.testing.assert_allclose(slide.axes[0].lines[0].get_ydata(), [0, 0, 2, 2, 0])
    assert slide.axes[0].get_ylim() == pytest.approx((4, 0))
    heat, heat_info = viz.render_slot_heatmaps(
        synthetic["thumbnail"], synthetic["slide_size"], synthetic["polygons"],
        synthetic["boxes"], synthetic["weights"], synthetic["visible"], synthetic["real"])
    assert heat_info["slot_count"] == 8
    assert heat_info["shared_color_range"] == [0.0, 0.75]
    assert len(heat.axes) == 9  # eight panels and one common colorbar
    assert {axis.images[0].get_clim() for axis in heat.axes[:8]} == {(0.0, 0.75)}
    image_data = heat.axes[0].images[0].get_array()
    assert np.ma.getmaskarray(image_data)[3, 3]
    for figure in (slide, heat):
        output = io.BytesIO()
        figure.savefig(output, format="png")
        assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_rejects_padding_in_plot_and_weights_not_conditioned(synthetic):
    with pytest.raises(ValueError, match="exclude padding"):
        viz.render_slot_heatmaps(
            synthetic["thumbnail"], synthetic["slide_size"], synthetic["polygons"],
            synthetic["boxes"], synthetic["weights"], np.ones(3, dtype=bool), synthetic["real"])
    bad_weights = synthetic["weights"].copy()
    bad_weights[:, :2] *= 0.5
    with pytest.raises(ValueError, match="conditioned"):
        viz.render_slot_heatmaps(
            synthetic["thumbnail"], synthetic["slide_size"], synthetic["polygons"],
            synthetic["boxes"], bad_weights, synthetic["visible"], synthetic["real"])
    bad_boxes = synthetic["boxes"].copy()
    bad_boxes[0] = [-1, 0, 2, 2]
    with pytest.raises(ValueError, match="inside"):
        viz.render_roi_overlay(
            synthetic["thumbnail"], synthetic["slide_size"], synthetic["polygons"],
            bad_boxes, synthetic["visible"], synthetic["real"])
    with pytest.raises(ValueError, match="too large"):
        viz.render_slot_heatmaps(
            np.zeros((1500, 1500, 3), dtype=np.uint8), synthetic["slide_size"],
            synthetic["polygons"], synthetic["boxes"], synthetic["weights"],
            synthetic["visible"], synthetic["real"])


def test_rejects_stretched_thumbnail_and_degenerate_roi(synthetic):
    # A thumbnail stretched to the wrong aspect ratio would shift drawn ROIs
    # and patch boxes; checking ratio cannot by itself verify image identity.
    with pytest.raises(ValueError, match="aspect ratio"):
        viz.render_roi_overlay(
            np.zeros((4, 10, 3), dtype=np.uint8), synthetic["slide_size"],
            synthetic["polygons"], synthetic["boxes"], synthetic["visible"], synthetic["real"])
    with pytest.raises(ValueError, match="aspect ratio"):
        viz.render_slot_heatmaps(
            np.zeros((10, 4, 3), dtype=np.uint8), synthetic["slide_size"],
            synthetic["polygons"], synthetic["boxes"], synthetic["weights"],
            synthetic["visible"], synthetic["real"])
    with pytest.raises(ValueError, match="nonzero area"):
        viz.render_roi_overlay(
            synthetic["thumbnail"], synthetic["slide_size"],
            [np.array([[0, 0], [1, 1], [2, 2]])], synthetic["boxes"],
            synthetic["visible"], synthetic["real"])


def test_overlap_and_top_gallery_show_shared_rows_without_normal_label():
    overlap = np.array([[1.0, 0.7], [0.7, 1.0]])
    fig, metadata = viz.render_overlap_matrix(overlap)
    assert metadata["color_range"] == [0.0, 1.0]
    assert "uniform" in fig.axes[0].get_title()
    with pytest.raises(ValueError, match="symmetric"):
        viz.render_overlap_matrix([[1.0, 0.7], [0.6, 1.0]])
    tiles = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    top = np.array([[0, 1], [1, 0]])
    real = np.array([True, True, False])
    gallery, metadata = viz.render_top_gallery(tiles, top, real, np.array([True, False, False]))
    assert metadata["distinct_shared_rows"] == 2
    assert "annotated tumor" in gallery.axes[0].get_title()
    assert "unknown" in gallery.axes[1].get_title()
    assert "normal" not in gallery.axes[1].get_title().lower()
    with pytest.raises(ValueError, match="never padding"):
        viz.render_top_gallery(tiles, np.array([[0, 2]]), real)
    missing, status = viz.render_top_gallery(None, None, None)
    assert missing is None
    assert status["status"] == "missing_patch_tiles"


def test_sparse_top_tiles_show_missing_without_inventing_images():
    fig, info = viz.render_top_gallery({0: np.zeros((4, 4, 3), dtype=np.uint8)},
                                       np.array([[0, 1], [1, 0]]),
                                       np.array([True, True, False]))
    assert info["status"] == "partial_missing_patch_tiles"
    assert info["missing_tile_panels"] == 2
    assert any(any(text.get_text() == "tile missing" for text in ax.texts) for ax in fig.axes)
    with pytest.raises(ValueError, match="real input rows"):
        viz.render_top_gallery({2: np.zeros((4, 4, 3), dtype=np.uint8)},
                               np.array([[0]]), np.array([True, True, False]))


def test_save_is_private_and_never_overwrites(monkeypatch, tmp_path):
    pretend_source = tmp_path / "slotspe_proof" / "roi_visualization.py"
    monkeypatch.setattr(viz, "__file__", str(pretend_source))
    fig, _ = viz.render_overlap_matrix([[1.0]])
    with pytest.raises(ValueError, match="local_evidence"):
        viz.save_figures({"overlap_matrix": fig}, tmp_path / "public")
    target = tmp_path / "local_evidence" / "generated_test"
    receipt = viz.save_figures({"overlap_matrix": fig, "top_gallery": None}, target)
    assert (target / "overlap_matrix.png").read_bytes().startswith(b"\x89PNG")
    assert receipt["missing"] == ["top_gallery"]
    with pytest.raises(FileExistsError):
        viz.save_figures({"overlap_matrix": fig}, target)


def test_pathway_heatmap_preserves_all_tokens_and_rejects_ambiguous_names():
    names = [f"pathway_{i}" for i in range(37)]
    weights = np.full((8, 37), 1 / 37)
    fig, info = viz.render_pathway_heatmap(weights, names, max_tick_labels=5)
    assert fig.axes[0].images[0].get_array().shape == (8, 37)
    assert info["pathway_count"] == 37
    assert info["displayed_tick_count"] <= 5
    assert info["single_gene_attribution"] is False
    with pytest.raises(ValueError, match="unique"):
        viz.render_pathway_heatmap(weights, ["duplicate"] * 37)
    with pytest.raises(ValueError, match="normalized"):
        viz.render_pathway_heatmap(weights * 2, names)
