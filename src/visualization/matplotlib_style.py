from __future__ import annotations

import matplotlib.pyplot as plt


def setup_plot_style() -> None:
    """Use a font that exists in standard Matplotlib installs.

    Figure text is intentionally English so the server does not need Chinese
    system fonts such as SimHei.
    """

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def setup_chinese_font() -> None:
    # Backward-compatible name used by existing plotting modules.
    setup_plot_style()

