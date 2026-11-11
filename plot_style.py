"""Times New Roman (or closest serif) and large, readable text for all GENCO / TPGT figures."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager

_FONT_REGISTERED = False

FONT_TICK = 16
FONT_LABEL = 18
FONT_TITLE = 20
FONT_LEGEND = 15
FONT_SUPTITLE = 20
FONT_CBAR = 16
FONT_ANNOTATE = 13


def _register_bundled_fonts() -> None:
    """Load optional ``fonts/*.ttf`` next to this module (e.g. Liberation Serif on Colab)."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    if fonts_dir.is_dir():
        for path in sorted(fonts_dir.glob("*.ttf")):
            font_manager.fontManager.addfont(str(path))
    _FONT_REGISTERED = True


def times_family() -> str:
    _register_bundled_fonts()
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        return "Times New Roman"
    for name in (
        "Liberation Serif",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "STIXGeneral",
        "DejaVu Serif",
    ):
        if name in available:
            return name
    return "serif"


def font_properties(size: float | None = None) -> font_manager.FontProperties:
    return font_manager.FontProperties(family=times_family(), size=size if size is not None else FONT_LABEL)


def apply_plot_style() -> None:
    """Set matplotlib rcParams for consistent typography (call once before creating figures)."""
    import matplotlib.pyplot as plt

    family = times_family()
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": family,
            "font.serif": [family, "Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
            "font.size": FONT_LABEL,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "figure.titlesize": FONT_SUPTITLE,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def style_axis(ax) -> None:
    tick_font = font_properties(FONT_TICK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_font)
    ax.xaxis.label.set_fontproperties(font_properties(FONT_LABEL))
    ax.yaxis.label.set_fontproperties(font_properties(FONT_LABEL))
    title = ax.get_title()
    if title:
        ax.title.set_fontproperties(font_properties(FONT_TITLE))


def style_colorbar(cbar) -> None:
    tick_font = font_properties(FONT_TICK)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(tick_font)
    label = cbar.ax.get_ylabel()
    if label:
        cbar.set_label(label, fontproperties=font_properties(FONT_CBAR))
