"""Display units: keep in/day and °F as they arrive; standardise raw model units."""

import numpy as np
import pytest
import xarray as xr

from weather_skills_plotting.display_units import (
    display_levels,
    display_units_agree,
    prepare_for_display,
    to_display_units,
)


def _ds(name, units, value, **attrs):
    times = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-01-08"))
    ds = xr.Dataset({name: (("time",), np.full(times.size, value, dtype=float))})
    ds = ds.assign_coords(time=times.astype("datetime64[ns]"))
    ds[name].attrs.update(units=units, **attrs)
    return ds


@pytest.mark.parametrize(
    ("name", "units", "value"),
    [
        ("precip", "inch day-1", 0.5),
        ("precip", "in/day", 0.5),
        ("t2m", "degree_Fahrenheit", 77.0),
        ("t2m", "degF", 77.0),
    ],
)
def test_requested_display_units_are_kept(name, units, value):
    out = to_display_units(_ds(name, units, value), [name])
    assert out[name].attrs["units"] == units
    assert float(out[name][0]) == pytest.approx(value)


def test_kept_precip_rate_gets_its_standard_name():
    out = to_display_units(_ds("precip", "inch day-1", 0.5), ["precip"])
    assert out["precip"].attrs["standard_name"] == "lwe_precipitation_rate"


@pytest.mark.parametrize(
    ("name", "units", "value", "expected_units", "expected"),
    [
        ("t2m", "K", 300.0, "degree_Celsius", 300.0 - 273.15),
        ("precip", "kg m-2 s-1", 1.0 / 86400.0, "mm day-1", 1.0),
        ("precip", "mm day-1", 12.7, "mm day-1", 12.7),
    ],
)
def test_raw_and_standard_units_take_the_standard_path(
    name, units, value, expected_units, expected
):
    out = to_display_units(_ds(name, units, value), [name])
    assert out[name].attrs["units"] == expected_units
    assert float(out[name][0]) == pytest.approx(expected)


def test_keep_false_is_plain_standardisation():
    out = to_display_units(_ds("t2m", "degF", 77.0), ["t2m"], keep=False)
    assert float(out["t2m"][0]) == pytest.approx(25.0)


def test_aggregated_inch_rate_becomes_an_inch_total():
    ds = _ds("precip", "inch day-1", 0.5, aggregation_period="1 day")
    out = prepare_for_display(ds, "precip")
    assert out["precip"].attrs["units"] == "inch"
    assert out["precip"].attrs["long_name"] == "Total precipitation"
    assert float(out["precip"][0]) == pytest.approx(0.5)
    mm = prepare_for_display(_ds("precip", "mm day-1", 3.0, aggregation_period="1 day"), "precip")
    assert mm["precip"].attrs["units"] == "mm"
    assert float(mm["precip"][0]) == pytest.approx(3.0)


def test_display_units_agree():
    f = _ds("t2m", "degF", 77.0)
    assert display_units_agree([(f, "t2m"), (_ds("t2m", "degree_Fahrenheit", 70.0), "t2m")])
    assert not display_units_agree([(f, "t2m"), (_ds("t2m", "K", 300.0), "t2m")])
    assert display_units_agree([(_ds("t2m", "K", 300.0), "t2m"), (_ds("t2m", "degC", 20.0), "t2m")])


def test_display_levels_convert_only_kept_units():
    bounds = [0, 1, 2, 5, 10, 25.4]
    inch = to_display_units(_ds("precip", "inch", 0.5), ["precip"])["precip"]
    assert display_levels(bounds, inch, "mm") == pytest.approx(
        [0, 0.0394, 0.0787, 0.197, 0.394, 1.0]
    )
    mm = to_display_units(_ds("precip", "mm", 5.0), ["precip"])["precip"]
    assert display_levels(bounds, mm, "mm") is bounds


def test_loaded_quantified_fields_keep_display_units():
    """Skills load Zarrs pint-quantified (no units attr): kept the same way."""
    f = to_display_units(_ds("t2m", "degF", 77.0).pint.quantify(), ["t2m"])
    assert float(f["t2m"][0].pint.magnitude) == pytest.approx(77.0)
    k = to_display_units(_ds("t2m", "K", 300.0).pint.quantify(), ["t2m"])
    assert float(k["t2m"][0].pint.magnitude) == pytest.approx(300.0 - 273.15)
    rate = _ds("precip", "inch day-1", 0.5, aggregation_period="1 day").pint.quantify()
    total = prepare_for_display(rate, "precip")
    assert total["precip"].attrs["units"] == "inch"
    assert float(total["precip"][0]) == pytest.approx(0.5)
