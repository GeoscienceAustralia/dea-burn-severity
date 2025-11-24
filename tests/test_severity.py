import numpy as np
import xarray as xr

from dea_burn_severity import severity


def _landcover(levels: list[list[int]]) -> xr.Dataset:
    data = xr.DataArray(
        np.array(levels, dtype=np.int32),
        dims=("y", "x"),
        coords={"y": range(len(levels)), "x": range(len(levels[0]))},
        name="level4",
    )
    return xr.Dataset({"level4": data})


def test_calculate_severity_handles_grass_vs_woody():
    landcover_ds = _landcover([[1, 3], [2, 14]])
    nbr = xr.DataArray(
        np.array([[0.5, 0.3], [0.05, 0.2]], dtype=np.float32),
        dims=("y", "x"),
        coords=landcover_ds.level4.coords,
    )

    result = severity.calculate_severity(nbr, landcover_ds, grass_classes=(3, 14))

    expected = np.array([[4, 2], [0, 2]], dtype=np.uint8)
    np.testing.assert_array_equal(result.values, expected)
    assert result.name == "severity"


def test_create_debug_mask_sets_all_flags(monkeypatch):
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")

    pre = xr.Dataset(
        {
            "nbart_red": xr.DataArray(np.ones((1, 1)), dims=("y", "x")),
            "oa_s2cloudless_mask": xr.DataArray([[2]], dims=("y", "x")),
            "oa_nbart_contiguity": xr.DataArray([[0]], dims=("y", "x")),
        }
    )
    post = xr.Dataset(
        {
            "oa_s2cloudless_mask": xr.DataArray(
                np.array([[[2]], [[2]]]), dims=("time", "y", "x"), coords={"time": times}
            ),
            "oa_nbart_contiguity": xr.DataArray(
                np.array([[[0]], [[0]]]), dims=("time", "y", "x"), coords={"time": times}
            ),
        }
    )

    def fake_calc_indices(stack, index, collection, drop):
        assert index == "MNDWI"
        data = xr.DataArray(
            np.array([[[0.5]], [[0.5]]]),
            dims=("time", "y", "x"),
            coords={"time": times},
            name="MNDWI",
        )
        return xr.Dataset({"MNDWI": data})

    monkeypatch.setattr(severity, "calculate_indices", fake_calc_indices)

    mask = severity.create_debug_mask(pre, post)

    assert mask.shape == (1, 1)
    assert mask.name == "debug_mask"
    assert int(mask.values[0, 0]) == 11111
