"""Plot a field in the units it arrives in when those are a display unit for it.

``to_standard_units`` turns every recognised air temperature into °C and every
precip rate into mm/day. Called unconditionally before plotting, that undid a
user's request: a Zarr already converted to in/day or °F was drawn in mm/day or
°C. The helpers here keep a field whose units are already a supported DISPLAY
unit for its quantity (``DISPLAY_UNITS``), and standardise everything else, so
raw model units (``K``, ``kg m-2 s-1``, ``m``, ``Pa``, ``mm h-1``) still become
°C / mm/day / mm / hPa exactly as before. Data already in standard units takes
the unchanged ``to_standard_units`` path.

Several fields drawn against one scale (an overlay, a verify grid, forecast vs
climate) only keep their units when they all agree; otherwise every one is
standardised, which is the previous behaviour.

Palettes defined in standard units (the nested precip mm classes) are
converted to the plotted unit with ``display_levels``, never the data.
"""

from __future__ import annotations

import numpy as np
from weather_skills_core.units import (
    STANDARD,
    classify_variable,
    convert_dataarray,
    convert_levels,
    precip_for_display,
    to_standard_units,
    units_equal,
    variable_units,
)

# Non-standard units a figure may show as they arrive, per ``STANDARD`` kind.
# The standard unit of each kind is always acceptable too (it takes the normal
# ``to_standard_units`` path). Anything else for a recognised kind is converted.
DISPLAY_UNITS = {
    "temp": ("degree_Fahrenheit", "delta_degree_Fahrenheit"),
    "precip": ("inch day-1",),
    "precip_amount": ("inch",),
}

# The amount unit a kept precip rate becomes when ``precip_for_display`` turns
# it into a period total (which the core always writes in mm).
_TOTAL_UNITS_OF_RATE = {"inch day-1": "inch"}


def _kind(da, name) -> str | None:
    return classify_variable(
        name or "", units=variable_units(da), standard_name=da.attrs.get("standard_name")
    )


def kept_display_units(da, name=None) -> str | None:
    """The ``DISPLAY_UNITS`` entry ``da`` is already in, or None."""
    units = variable_units(da)  # pint units of a loaded (quantified) field, else the attr
    if not units:
        return None
    kind = _kind(da, name if name is not None else da.name)
    for candidate in DISPLAY_UNITS.get(kind, ()):
        if units_equal(units, candidate):
            return candidate
    return None


def to_display_units(ds, variables, *, keep: bool = True):
    """``to_standard_units`` except for variables already in a ``DISPLAY_UNITS`` unit.

    A kept variable's values and ``units`` are untouched; it gets its kind's CF
    ``standard_name`` stamped, as ``to_standard_units`` would. ``keep=False``
    is plain ``to_standard_units``.
    """
    names = list(variables)
    if not keep:
        return to_standard_units(ds, variables=names)
    out = ds
    for name in names:
        da = out[name] if name in out.data_vars else None
        if da is None or kept_display_units(da, name) is None:
            # Unknown names raise inside to_standard_units, as before.
            out = to_standard_units(out, variables=[name])
            continue
        standard_name = STANDARD[_kind(da, name)]["standard_name"]
        if standard_name and da.attrs.get("standard_name") != standard_name:
            out = out.copy()
            out[name].attrs = {**da.attrs, "standard_name": standard_name}
    return out


def precip_total_for_display(ds, name):
    """``precip_for_display``, keeping a kept rate's length unit (in/day → in)."""
    if name not in ds.data_vars:
        return precip_for_display(ds, name)
    source = kept_display_units(ds[name], name)
    out = precip_for_display(ds, name)
    target = _TOTAL_UNITS_OF_RATE.get(source)
    if out is ds or target is None:
        return out
    da = out[name]
    if not units_equal(variable_units(da), STANDARD["precip_amount"]["units"]):
        return out
    converted, _ = convert_dataarray(da, target)
    out = out.copy(deep=False)
    out[name] = converted
    out[name].attrs = {**da.attrs, "units": target}
    return out


def prepare_for_display(ds, name, *, keep: bool = True):
    """Units for the figure, then the in-memory precip total (the plotters' one call)."""
    return precip_total_for_display(to_display_units(ds, [name], keep=keep), name)


def display_units_agree(pairs) -> bool:
    """True when every ``(ds, variable)`` keeps the same units after ``to_display_units``.

    Use it before drawing several fields against one scale: when it is False,
    pass ``keep=False`` so they are all standardised (and so comparable).
    Variables without units do not take part.
    """
    seen = []
    for ds, name in pairs:
        if name not in ds.data_vars:
            continue
        units = variable_units(to_display_units(ds, [name])[name])
        if units:
            seen.append(units)
    return all(units_equal(seen[0], other) for other in seen[1:])


def display_levels(levels, da, palette_units: str):
    """Palette breakpoints written in ``palette_units``, in the units ``da`` is plotted in.

    Only converts when ``da`` is in a kept non-standard ``DISPLAY_UNITS`` unit,
    so a field in standard units gets its levels back unchanged. Converted
    levels are rounded to 3 significant figures for readable colorbar ticks
    (unless that would merge two levels).
    """
    if levels is None:
        return levels
    target = kept_display_units(da)
    if target is None or units_equal(target, palette_units):
        return levels
    converted = convert_levels(levels, palette_units, target)
    rounded = np.array([float(f"{v:.3g}") for v in converted])
    if np.all(np.diff(rounded) > 0):
        converted = rounded
    return [float(v) for v in converted]
