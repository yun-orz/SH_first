from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


def configure_chinese_font() -> None:
    """优先注册 Windows 中文字体；找不到时保留 Matplotlib 默认字体。"""
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            mpl.rcParams["font.family"] = family
            mpl.rcParams["axes.unicode_minus"] = False
            return

