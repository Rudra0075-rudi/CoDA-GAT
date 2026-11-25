"""Shared look for the CoDA-GAT result figures.

Clean matplotlib: sans-serif type, left/bottom axes only, no grid, frameless
legends, one semantic palette used the same way in every figure.
Drawn at the RSOS text width (135 mm) so printed type is 6.5-8 pt.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
TEXTW = 135 * MM            # RSOS text width in inches

# semantic palette
BLUE = "#0F4D92"            # CoDA-GAT / STRING graph (method under test)
BLUE2 = "#3775BA"           # graph variants
BLUE_L = "#9DBEE3"
BLUE_XL = "#DCE8F5"
GREEN = "#3E8E4A"           # tabular comparison models
GREEN_L = "#9FD49F"
GREEN_XL = "#DDF3DE"
RED = "#B64342"             # failures, flags, shortfalls
RED_L = "#E9A6A1"
RED_XL = "#F6CFCB"
GREY_D = "#3A3A3A"
GREY = "#767676"
GREY_L = "#BDBDBD"
GREY_XL = "#ECECEC"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
GOLD = "#C99A06"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 7.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.2,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "errorbar.capsize": 0,
    "mathtext.default": "regular",
})


def ref_line(ax, x=None, y=None, **kw):
    style = dict(color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    style.update(kw)
    if x is not None:
        ax.axvline(x, **style)
    if y is not None:
        ax.axhline(y, **style)


def zero_line(ax, orient="h"):
    (ax.axhline if orient == "h" else ax.axvline)(0, color=INK, lw=0.7, zorder=1)


def only_bottom(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def only_left(ax):
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
