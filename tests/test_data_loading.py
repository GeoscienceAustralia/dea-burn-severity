from types import SimpleNamespace

import numpy as np
import xarray as xr

from dea_burn_severity.configuration import DEFAULT_CONFIG_DICT, RuntimeConfig
from dea_burn_severity import data_loading
from dea_burn_severity.data_loading import load_ard_with_fallback, load_baseline_stack


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig.from_dict(DEFAULT_CONFIG_DICT)


def _nonempty_dataset(value: float = 1.0) -> xr.Dataset:
    time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    data = xr.DataArray(
        np.full((1, 1, 1), value),
        coords={"time": time, "y": [0], "x": [0]},
        dims=("time", "y", "x"),
        name="nbart_green",
    )
    return xr.Dataset({"nbart_green": data})


def test_load_ard_with_fallback_returns_on_first_success(monkeypatch):
    calls: list[float] = []

    def fake_load_ard(**kwargs):
        calls.append(kwargs["min_gooddata"])
        if kwargs["min_gooddata"] < 0.95:
            return _nonempty_dataset()
        return xr.Dataset()

    monkeypatch.setattr(data_loading, "load_ard", fake_load_ard)

    ds = load_ard_with_fallback(
        dc=SimpleNamespace(),
        gpgon=SimpleNamespace(),
        time=("start", "end"),
        config=_runtime_config(),
        min_gooddata_thresholds=(0.99, 0.9),
    )

    assert calls == [0.99, 0.9]
    assert ds.time.size == 1


def test_load_baseline_stack_builds_relaxed_composite(monkeypatch):
    calls: list[float] = []

    empty = xr.Dataset()
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    baseline_values = xr.DataArray(
        np.array([[[np.nan]], [[5.0]]]),
        coords={"time": times, "y": [0], "x": [0]},
        dims=("time", "y", "x"),
        name="nbart_green",
    )
    relaxed_ds = xr.Dataset({"nbart_green": baseline_values})

    responses = [empty, relaxed_ds]

    def fake_load_ard(**kwargs):
        calls.append(kwargs.get("min_gooddata"))
        return responses.pop(0)

    monkeypatch.setattr(data_loading, "load_ard", fake_load_ard)

    baseline, composite = load_baseline_stack(
        dc=SimpleNamespace(),
        gpgon=SimpleNamespace(),
        time=("start", "end"),
        config=_runtime_config(),
    )

    assert calls == [0.99, 0.5]
    assert baseline.time.size == 2
    assert "time" not in composite.dims
    assert composite.nbart_green.values.item() == 5.0
