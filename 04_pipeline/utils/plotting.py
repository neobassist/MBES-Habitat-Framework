"""Shared publication typography and projected-map cartography helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence


PREFERRED_SANS_FONTS: tuple[str, ...] = (
    "Arial",
    "Helvetica",
    "DejaVu Sans",
)


def configure_publication_font(
    preferred: Sequence[str] = PREFERRED_SANS_FONTS,
) -> str:
    """Configure editable publication text and return the resolved font family.

    Arial is preferred, with Helvetica and DejaVu Sans as portable fallbacks.
    The returned name is Matplotlib's actual resolved font, not merely the
    requested family.
    """

    import matplotlib as mpl
    from matplotlib import font_manager

    families = [str(value) for value in preferred if str(value).strip()]
    if not families:
        families = list(PREFERRED_SANS_FONTS)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = families
    properties = font_manager.FontProperties(family=families)
    resolved_path = font_manager.findfont(properties, fallback_to_default=True)
    return str(font_manager.FontProperties(fname=resolved_path).get_name())


def _bounds_tuple(bounds: Any) -> tuple[float, float, float, float]:
    """Normalize rasterio-style bounds or a four-value sequence."""

    if all(hasattr(bounds, name) for name in ("left", "bottom", "right", "top")):
        return (
            float(bounds.left),
            float(bounds.bottom),
            float(bounds.right),
            float(bounds.top),
        )
    left, bottom, right, top = bounds
    return float(left), float(bottom), float(right), float(top)


def nice_scale_length(width_m: float, target_fraction: float = 0.20) -> float:
    """Return a 1/2/5 x 10^n scale length near 20% of map width."""

    width = float(width_m)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Map width must be a positive finite projected distance.")
    target = width * float(target_fraction)
    exponent = math.floor(math.log10(target))
    candidates = [
        multiplier * 10.0**power
        for power in range(exponent - 1, exponent + 2)
        for multiplier in (1.0, 2.0, 5.0)
    ]
    in_range = [value for value in candidates if 0.15 <= value / width <= 0.25]
    pool = in_range or candidates
    return min(pool, key=lambda value: abs(value - target))


def add_scale_bar(
    ax: Any,
    bounds: Any,
    *,
    location: tuple[float, float] = (0.08, 0.08),
) -> float:
    """Draw a coordinate-derived metric scale bar and return its length."""

    left, bottom, right, top = _bounds_tuple(bounds)
    width = right - left
    height = top - bottom
    scale_length = nice_scale_length(width)
    x0 = left + width * float(location[0])
    y0 = bottom + height * float(location[1])
    ax.plot(
        [x0, x0 + scale_length],
        [y0, y0],
        color="black",
        linewidth=2.6,
        solid_capstyle="butt",
        zorder=20,
    )
    label = f"{scale_length / 1000:g} km" if scale_length >= 1000 else f"{scale_length:g} m"
    ax.text(
        x0 + scale_length / 2,
        y0 + height * 0.025,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=21,
    )
    return scale_length


def add_north_arrow(ax: Any) -> None:
    """Draw a compact locator-style grid/map-north symbol.

    The concave white arrow follows the visual proportions of the approved
    StudyArea_Zoom locator map while remaining legible over either light or
    dark raster values.  Coordinates are axes-relative so the symbol stays in
    the true upper-right cartographic position for every map panel.
    """

    from matplotlib import patheffects
    from matplotlib.patches import Polygon

    center_x = 0.958
    tip_y = 0.925
    base_y = 0.855
    notch_y = 0.881
    half_width = 0.021
    arrow = Polygon(
        [
            (center_x, tip_y),
            (center_x + half_width, base_y),
            (center_x, notch_y),
            (center_x - half_width, base_y),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.25,
        joinstyle="miter",
        clip_on=True,
        zorder=22,
    )
    ax.add_patch(arrow)
    ax.plot(
        [center_x, center_x],
        [tip_y, notch_y],
        transform=ax.transAxes,
        color="black",
        linewidth=0.55,
        clip_on=True,
        zorder=23,
    )
    label = ax.text(
        center_x,
        0.948,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="white",
        fontsize=8.5,
        fontweight="bold",
        clip_on=True,
        zorder=24,
    )
    label.set_path_effects(
        [patheffects.withStroke(linewidth=1.35, foreground="black")]
    )


def add_map_cartography(ax: Any, bounds: Any) -> float:
    """Add the shared north arrow and scale bar to a projected map axis."""

    scale_length = add_scale_bar(ax, bounds)
    add_north_arrow(ax)
    return scale_length


__all__ = (
    "PREFERRED_SANS_FONTS",
    "add_map_cartography",
    "add_north_arrow",
    "add_scale_bar",
    "configure_publication_font",
    "nice_scale_length",
)
