"""No WSI or patient data is read by these tests."""
import sys
import types

import numpy as np
import pytest
from PIL import Image

from slotspe_proof.roi_wsi_assets import extract_slide_assets, open_slide_assets


class FakeReader:
    dimensions = (100, 80)

    def __init__(self):
        self.calls = []
        self.closed = False

    def get_thumbnail(self, size):
        self.calls.append(("thumbnail", size))
        return Image.new("RGBA", (50, 40), (10, 20, 30, 255))

    def read_region(self, location, level, size):
        self.calls.append((location, level, size))
        return Image.new("RGBA", size, (100, 0, 0, 128))

    def close(self):
        self.closed = True


@pytest.fixture
def args():
    return {"slide_id": "slide-A", "slide_metadata": {
        "coordinate_space": "level0", "mapping_verified": True,
        "mapping_sha256": "ab" * 32, "feature_count": 3,
        "width": 100, "height": 80,
    }, "patches": [
        {"token_index": 0, "is_real": True, "slide_id": "slide-A", "feature_row": 1, "box": [10, 20, 20, 10]},
        {"token_index": 1, "is_real": True, "slide_id": "slide-B", "feature_row": 0, "box": [1, 1, 4, 4]},
        {"token_index": 2, "is_real": False, "slide_id": None, "feature_row": None, "box": None},
    ], "requested_rows": [0, 1, 2], "thumbnail_max_size": 64, "tile_size": (16, 8)}


def test_sparse_exact_slide_and_padding(args):
    reader = FakeReader()
    out = extract_slide_assets(reader, **args)
    assert out["slide_size"] == (100, 80)
    assert out["thumbnail"].shape == (40, 50, 3)
    assert out["tiles"][0].shape == (8, 16, 3)
    assert out["tiles"][0].dtype == np.uint8
    assert out["missing_rows"] == {1: "different_slide", 2: "padding"}
    assert out["identity_independently_verified_by_this_helper"] is False
    assert reader.calls == [("thumbnail", (64, 64)), ((10, 20), 0, (20, 10))]
    assert np.all(out["tiles"][0][0, 0] == [177, 127, 127])


@pytest.mark.parametrize("box", [[99, 0, 2, 4], [-1, 0, 2, 4], [0, 0, 0, 4], [0., 0, 2, 4], [0, 0, 2]])
def test_invalid_box_fails_before_reading(args, box):
    args["patches"][0]["box"] = box
    reader = FakeReader()
    with pytest.raises(ValueError):
        extract_slide_assets(reader, **args)
    assert not reader.calls


def test_rejects_bad_mapping_and_reader_dimensions(args):
    args["slide_metadata"]["mapping_verified"] = False
    with pytest.raises(ValueError, match="mapping"):
        extract_slide_assets(FakeReader(), **args)
    args["slide_metadata"]["mapping_verified"] = True
    args["slide_metadata"]["width"] = 99
    with pytest.raises(ValueError, match="dimensions"):
        extract_slide_assets(FakeReader(), **args)


def test_rejects_duplicate_and_out_of_range_requested_tokens(args):
    args["requested_rows"] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        extract_slide_assets(FakeReader(), **args)
    args["requested_rows"] = [3]
    with pytest.raises(ValueError, match="outside"):
        extract_slide_assets(FakeReader(), **args)


def test_numpy_token_indices_are_supported(args):
    args["requested_rows"] = np.asarray([0, 1], dtype=np.int64)
    out = extract_slide_assets(FakeReader(), **args)
    assert list(out["tiles"]) == [0]
    assert out["missing_rows"] == {1: "different_slide"}


def test_rejects_wrong_manifest_order(args):
    args["patches"][1]["token_index"] = 2
    with pytest.raises(ValueError, match="order"):
        extract_slide_assets(FakeReader(), **args)


def test_open_slide_lazy_and_closes_even_if_invalid(monkeypatch, args):
    reader = FakeReader()
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=lambda path: reader))
    out = open_slide_assets("synthetic.fake", **args)
    assert out["tiles"][0].shape == (8, 16, 3)
    assert reader.closed
    another = FakeReader()
    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=lambda path: another))
    args["patches"][0]["box"] = [99, 0, 2, 4]
    with pytest.raises(ValueError):
        open_slide_assets("synthetic.fake", **args)
    assert another.closed
