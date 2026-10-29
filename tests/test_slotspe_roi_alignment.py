"""Independent synthetic geometry tests; no patient inputs or model calls."""
import copy

import pytest

pytest.importorskip("shapely")
from slotspe_proof.roi_alignment import align_patch_annotations


def polygon(points, holes=()):
    return {"type": "Polygon", "coordinates": [points, *holes]}


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def setup():
    slides = {
        "slide-A": {"feature_count": 4, "mapping_verified": True,
                    "mapping_sha256": "a" * 64, "coordinate_space": "level0",
                    "width": 100, "height": 100},
        "slide-B": {"feature_count": 1, "mapping_verified": True,
                    "mapping_sha256": "b" * 64},
    }
    patches = [
        {"token_index": 0, "is_real": True, "slide_id": "slide-A", "feature_row": 0,
         "box": [0, 0, 100, 100]},
        {"token_index": 1, "is_real": True, "slide_id": "slide-A", "feature_row": 1,
         "box": [75, 0, 10, 10]},
        {"token_index": 2, "is_real": True, "slide_id": "slide-A", "feature_row": 2,
         "box": [80, 0, 20, 20]},
        {"token_index": 3, "is_real": True, "slide_id": "slide-B", "feature_row": 0,
         "box": None},
        {"token_index": 4, "is_real": False, "slide_id": None, "feature_row": None,
         "box": None},
    ]
    annotations = {"slide-A": {"coordinate_space": "level0", "geometries": [
        polygon(rectangle(0, 0, 60, 100)),
        polygon(rectangle(40, 0, 80, 100)),
    ]}}
    return patches, slides, annotations


def test_overlap_union_edge_touch_slide_unknown_and_padding():
    patches, slides, annotations = setup()
    actual = align_patch_annotations(patches, slides, annotations, expected_token_count=5)
    assert actual["fractions"] == pytest.approx([0.8, 0.5, 0.0, None, None], nan_ok=True)
    assert actual["real_mask"] == [True, True, True, True, False]
    assert actual["tumor_mask"] == [True, True, False, False, False]
    assert actual["status"] == ["tumor", "tumor", "unknown", "unknown", "padding"]
    assert actual["sensitivity"]["0.75"] == [True, False, False, False, False]
    assert actual["counts"] == {"tokens": 5, "real": 4, "padding": 1, "tumor": 2, "unknown": 2}
    assert actual["mapping_evidence_independently_verified"] is False


def test_hole_is_not_filled_and_unannotated_slide_is_not_negative():
    patches, slides, annotations = setup()
    patches[0]["box"] = [0, 0, 100, 100]
    annotations["slide-A"]["geometries"] = [polygon(rectangle(0, 0, 100, 100), [rectangle(25, 25, 75, 75)])]
    actual = align_patch_annotations(patches, slides, annotations)
    assert actual["fractions"][0] == 0.75
    assert actual["status"][3] == "unknown"
    assert actual["fractions"][3] is None


def test_nonmatching_slide_id_is_unknown_and_exact_id_required():
    patches, slides, annotations = setup()
    slides["slide-A-OTHER"] = {"feature_count": 1, "mapping_verified": True, "mapping_sha256": "c" * 64}
    patches[3]["slide_id"] = "slide-A-OTHER"
    actual = align_patch_annotations(patches, slides, annotations)
    assert actual["fractions"][3] is None
    assert actual["status"][3] == "unknown"
    annotations["slide-a"] = annotations.pop("slide-A")
    with pytest.raises(ValueError, match="absent"):
        align_patch_annotations(patches, slides, annotations)


@pytest.mark.parametrize("case", ["token_order", "duplicate_row", "no_coordinates", "out_of_bounds", "bad_padding", "missing_field", "invalid_hash", "unconfirmed_mapping", "mismatched_coordinate_space", "count_mismatch", "no_real", "overflow_area", "underflow_area"])
def test_provenance_and_coordinate_contract_fails_closed(case):
    patches, slides, annotations = setup()
    kwargs = {}
    if case == "token_order":
        patches[0]["token_index"] = 1
    elif case == "duplicate_row":
        patches[1]["feature_row"] = 0
    elif case == "no_coordinates":
        patches[0]["box"] = None
    elif case == "out_of_bounds":
        patches[0]["box"] = [90, 0, 20, 20]
    elif case == "bad_padding":
        patches[4]["slide_id"] = "slide-A"
    elif case == "missing_field":
        del patches[3]["box"]
    elif case == "invalid_hash":
        slides["slide-A"]["mapping_sha256"] = "not-a-hash"
    elif case == "unconfirmed_mapping":
        slides["slide-A"]["mapping_verified"] = False
    elif case == "mismatched_coordinate_space":
        annotations["slide-A"]["coordinate_space"] = "level1"
    elif case == "count_mismatch":
        kwargs["expected_token_count"] = 6
    elif case == "no_real":
        patches = [patches[4]]
        patches[0]["token_index"] = 0
    elif case == "overflow_area":
        patches[0]["box"] = [0, 0, 1e200, 1e200]
    elif case == "underflow_area":
        patches[0]["box"] = [0, 0, 1e-200, 1e-200]
    with pytest.raises(ValueError):
        align_patch_annotations(patches, slides, annotations, **kwargs)


@pytest.mark.parametrize("case", ["unclosed", "self_intersection", "outside_slide", "empty", "nonfinite", "three_dimensions"])
def test_invalid_geometry_is_rejected(case):
    patches, slides, annotations = setup()
    if case == "unclosed":
        ring = [[0, 0], [10, 0], [10, 10], [0, 10]]
    elif case == "self_intersection":
        ring = [[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]]
    elif case == "outside_slide":
        ring = rectangle(-1, 0, 10, 10)
    elif case == "empty":
        annotations["slide-A"]["geometries"] = []
        ring = None
    elif case == "nonfinite":
        ring = rectangle(0, 0, float("nan"), 10)
    else:
        ring = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 0, 0]]
    if ring is not None:
        annotations["slide-A"]["geometries"] = [polygon(ring)]
    with pytest.raises(ValueError):
        align_patch_annotations(patches, slides, annotations)


def test_inputs_are_not_mutated():
    patches, slides, annotations = setup()
    originals = copy.deepcopy((patches, slides, annotations))
    align_patch_annotations(patches, slides, annotations)
    assert (patches, slides, annotations) == originals


def test_multipolygon_components_supported():
    patches, slides, annotations = setup()
    annotations["slide-A"]["geometries"] = [{"type": "MultiPolygon", "coordinates": [
        [rectangle(0, 0, 20, 100)], [rectangle(80, 0, 100, 100)]]}]
    actual = align_patch_annotations(patches, slides, annotations)
    assert actual["fractions"][0] == pytest.approx(0.4)
    assert not actual["tumor_mask"][0]
