"""Generated HDF5 layouts used by the shared real-input cache preparer."""
import numpy as np
import pytest

from survot_rank.research.slotspe_proof.patient import _h5_feature_array


@pytest.mark.parametrize("leading", [False, True])
def test_uni2h_feature_layout_keeps_all_patch_rows(tmp_path, leading):
    h5py = pytest.importorskip("h5py")
    expected = np.arange(35, dtype=np.float32).reshape(7, 5)
    path = tmp_path / "generated_features.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=expected[None] if leading else expected)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(_h5_feature_array(handle), expected)


@pytest.mark.parametrize("shape", [(2, 7, 5), (5,), (1, 2, 3, 4)])
def test_ambiguous_feature_layout_is_rejected(tmp_path, shape):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "generated_bad_features.h5", "w") as handle:
        handle.create_dataset("features", data=np.zeros(shape, dtype=np.float32))
        with pytest.raises(ValueError, match="HDF5 features"):
            _h5_feature_array(handle)
